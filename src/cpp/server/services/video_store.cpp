#include "lemon/server/video_store.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>

#include <lemon/utils/aixlog.hpp>

#include "lemon/utils/path_utils.h"

namespace lemon {
namespace video_store {

namespace fs = std::filesystem;

namespace {
const std::string kPrefix = "video-";
constexpr size_t kHexLength = 32;

std::mutex& reservation_mutex() {
    static std::mutex m;
    return m;
}

// Clip name -> the job it was reserved for.
std::map<std::string, std::string>& reserved() {
    static std::map<std::string, std::string> names;
    return names;
}

// Raw base64, or a data URL as sd-server also accepts.
bool is_base64_image(const std::string& value) {
    std::string_view data(value);
    if (data.rfind("data:image/", 0) == 0) {
        const auto comma = data.find(";base64,");
        if (comma == std::string_view::npos) return false;
        data.remove_prefix(comma + 8);
    }
    if (data.size() < 4 || data.size() % 4 != 0) return false;
    const auto pad = data.find('=');
    const size_t body = pad == std::string_view::npos ? data.size() : pad;
    if (data.size() - body > 2) return false;
    const auto is_char = [](unsigned char c) {
        return std::isalnum(c) || c == '+' || c == '/';
    };
    return std::all_of(data.begin(), data.begin() + body, is_char) &&
           std::all_of(data.begin() + body, data.end(), [](char c) { return c == '='; });
}
} // namespace

fs::path dir() {
    return utils::path_from_utf8(utils::get_cache_dir()) / "videos";
}

std::string new_file_name() {
    std::random_device device;
    std::seed_seq seed{device(), device(), device(), device()};
    std::mt19937_64 engine(seed);
    std::ostringstream name;
    name << kPrefix << std::hex;
    for (int i = 0; i < 2; ++i) {
        name.width(16);
        name.fill('0');
        name << engine();
    }
    return name.str();
}

bool is_valid_file_name(const std::string& name) {
    if (name.size() != kPrefix.size() + kHexLength || name.compare(0, kPrefix.size(), kPrefix) != 0)
        return false;
    return std::all_of(name.begin() + kPrefix.size(), name.end(),
                       [](unsigned char c) { return std::isxdigit(c) && !std::isupper(c); });
}

void write(const std::string& name, const std::string& bytes) {
    if (!is_valid_file_name(name)) throw std::runtime_error("invalid video file name");
    std::error_code ec;
    fs::create_directories(dir(), ec);
    std::ofstream out(dir() / name, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("could not write the video to " + utils::path_to_utf8(dir()));
}

bool read(const std::string& name, std::string& bytes) {
    if (!is_valid_file_name(name)) return false;
    std::ifstream in(dir() / name, std::ios::binary);
    if (!in) return false;
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

bool exists(const std::string& name) {
    std::error_code ec;
    return is_valid_file_name(name) && fs::exists(dir() / name, ec);
}

std::string reserve_file_name(const std::string& preferred, const std::string& job_id) {
    std::lock_guard<std::mutex> lock(reservation_mutex());
    std::string name = preferred;
    while (!is_valid_file_name(name) || reserved().count(name) || exists(name)) name = new_file_name();
    reserved()[name] = job_id;
    return name;
}

void release(const std::string& name) {
    std::lock_guard<std::mutex> lock(reservation_mutex());
    reserved().erase(name);
}


void clean_up(const std::function<CleanupSnapshot()>& snapshot) {
    std::lock_guard<std::mutex> lock(reservation_mutex());
    const CleanupSnapshot now = snapshot();
    for (auto it = reserved().begin(); it != reserved().end();) {
        const bool published = now.referenced.count(it->first) > 0;
        const bool job_gone = !now.live_jobs.count(it->second);
        it = (published || job_gone) ? reserved().erase(it) : std::next(it);
    }
    std::error_code ec;
    for (fs::directory_iterator it(dir(), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = utils::path_to_utf8(it->path().filename());
        if (!is_valid_file_name(name) || now.referenced.count(name) || reserved().count(name)) continue;
        std::error_code rm;
        fs::remove(it->path(), rm);
        if (rm) LOG(WARNING, "Videos") << "Could not remove " << name << ": " << rm.message() << std::endl;
    }
}

std::string normalize_request(nlohmann::json& request) {
    struct IntegerField {
        const char* name;
        int64_t min;
        int64_t max;
    };
    const IntegerField integers[] = {
        {"video_frames", 1, kMaxVideoFrames},
        {"fps", 1, kMaxFps},
        {"width", 1, kMaxDimension},
        {"height", 1, kMaxDimension},
        {"steps", 1, kMaxSteps},
        {"seed", std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()},
    };
    // 2^63 is exact as a double, while INT64_MAX is not: a double compared against
    // (double)INT64_MAX would let 2^63 itself through to an undefined cast.
    constexpr double kTwoPow63 = 9223372036854775808.0;
    for (const auto& field : integers) {
        if (!request.contains(field.name) || request[field.name].is_null()) continue;
        auto& value = request[field.name];
        const std::string out_of_range = std::string("'") + field.name + "' is out of range";
        if (!value.is_number()) return std::string("'") + field.name + "' must be an integer";
        int64_t number = 0;
        if (value.is_number_unsigned()) {
            const uint64_t unsigned_number = value.get<uint64_t>();
            if (unsigned_number > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
                return out_of_range;
            number = static_cast<int64_t>(unsigned_number);
        } else if (value.is_number_integer()) {
            number = value.get<int64_t>();
        } else {
            const double floating = value.get<double>();
            if (!std::isfinite(floating) || std::floor(floating) != floating)
                return std::string("'") + field.name + "' must be an integer";
            if (floating < -kTwoPow63 || floating >= kTwoPow63) return out_of_range;
            number = static_cast<int64_t>(floating);
        }
        if (number < field.min || number > field.max) return out_of_range;
        value = number;
    }
    struct NumberField {
        const char* name;
        double min;
        double max;
    };
    const NumberField numbers[] = {
        {"duration", 0, kMaxDuration},
        {"cfg_scale", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
        {"flow_shift", -std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
        {"vace_strength", 0, std::numeric_limits<double>::max()},
    };
    for (const auto& field : numbers) {
        if (!request.contains(field.name) || request[field.name].is_null()) continue;
        const auto& value = request[field.name];
        if (!value.is_number() || !std::isfinite(value.get<double>()))
            return std::string("'") + field.name + "' must be a number";
        if (value.get<double>() < field.min || value.get<double>() > field.max)
            return std::string("'") + field.name + "' is out of range";
    }
    if (request.contains("control_frames") && !request["control_frames"].is_null()) {
        const auto& frames = request["control_frames"];
        if (!frames.is_array() || frames.empty())
            return "'control_frames' must be a non-empty array of base64 images";
        if (frames.size() > static_cast<size_t>(kMaxVideoFrames))
            return "'control_frames' has more than " + std::to_string(kMaxVideoFrames) + " frames";
        size_t total = 0;
        for (const auto& frame : frames) {
            if (!frame.is_string() || !is_base64_image(frame.get_ref<const std::string&>()))
                return "each of 'control_frames' must be a base64 image or an image data URL";
            total += frame.get_ref<const std::string&>().size();
            if (total > kMaxControlFramesBytes) return "'control_frames' is larger than 256 MiB";
        }
    }
    return "";
}

bool is_video_job(const nlohmann::json& job) {
    if (!job.is_object() || !job.contains("steps") || !job["steps"].is_array() ||
        job["steps"].size() != 1) {
        return false;
    }
    const auto& step = job["steps"][0];
    return step.is_object() && step.value("op", std::string()) == "generate_video";
}

std::string video_status(const nlohmann::json& job) {
    const std::string status = job.value("status", std::string());
    if (status == "paused" && is_video_job(job) &&
        job["steps"][0].value("status", std::string()) == "completed") {
        return "completed";
    }
    if (status != "interrupted") return status;
    return job.value("error", std::string()).empty() ? "canceled" : "failed";
}

} // namespace video_store
} // namespace lemon
