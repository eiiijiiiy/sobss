# pragma once

#include <iostream>
#include <string>
#include <map>
#include <filesystem>

// rapidjson includes
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/writer.h>

namespace rj = rapidjson;
namespace fs = std::filesystem;

using namespace std;