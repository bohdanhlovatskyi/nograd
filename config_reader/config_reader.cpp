// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include "config_reader.hpp"
#include "string_tools.h"

std::string trim(std::string arg) {
    constexpr auto is_space_priv = [](auto a) { return std::isspace(a); };

    auto last_nonspace = std::find_if_not(arg.rbegin(), arg.rend(), is_space_priv ).base();
    if(last_nonspace != arg.end())
        arg.erase(last_nonspace, arg.end());

    auto first_nonspace = std::find_if_not(arg.begin(), arg.end(), is_space_priv );
    arg.erase(arg.begin(), first_nonspace);

    return arg;
}

template<typename T>
T conf::get_option_from_map(const std::string& option_name) const {
    if( options_map.count(option_name) == 0 ) {
        throw option_not_found_error("Option not found: " + option_name); // Тут можу запитатися -- чого воно працює?
    }

    try {
        auto elem_itr = options_map.find(option_name);

        if( elem_itr != options_map.end() )
            return from_str<T>(elem_itr->second);
        else
            throw wrong_option_arg_error{"Option " + option_name + " not found"};

    } catch( std::runtime_error& ex ){
        throw wrong_option_arg_error{ex.what()};
    }
}

conf::parse_string_ret_t
conf::parse_string(std::string arg) { // Note: we need copy here -- let compiler create it for us
    constexpr char separator = '='; // Just for the readability
    constexpr char commenter = '#';

    auto comment_pos = arg.find(commenter);
    if (comment_pos != std::string::npos)
        arg.erase(comment_pos);

    auto sep_pos = arg.find(separator);
    if (sep_pos == std::string::npos){
        return parse_string_ret_t{trim(arg), std::nullopt};
    }

    auto left_part = arg.substr(0, sep_pos);
    auto right_part = arg.substr(sep_pos+1, std::string::npos );

    return parse_string_ret_t{trim(left_part), trim(right_part)};
}

void conf::file_to_options_map(std::istream& cf){
    std::string line;

    while( std::getline(cf, line) ){
        auto pr = parse_string(line);

        if(pr.first.empty()) {
            if(!pr.second)
                continue;
            else
                throw wrong_option_line_error{"Wrong config line -- no option name: " + line}; // "=..."
        } else if(!pr.second){
            throw wrong_option_line_error{"Wrong config line -- no '=': " + line}; // "abc" -- no '='
        } else if(pr.second->empty()){
            throw wrong_option_arg_error{"Wrong config option value: " + line}; // "key="
        }
        if( options_map.count(pr.first) ){
            throw option_duplicated_error{"Duplicated option name: " + pr.first + " = "
                                          + *pr.second + "; prev. val: " + options_map[pr.first] };
        }
        options_map[pr.first] = *pr.second;
    }
}

std::filesystem::path kDataRoot;
size_t kTrainBatchSize;
size_t kTestBatchSize;
size_t kNumberOfEpochs;
size_t kLogInterval;
void conf::read_conf(std::istream& cf)
{
    // allows to create c++ string literal as ""s
    using namespace std::literals::string_literals;

    file_to_options_map(cf);
    kDataRoot = get_option_from_map<decltype(kDataRoot)>("kDataRoot"s);
    kTrainBatchSize = get_option_from_map<decltype(kTrainBatchSize)>("kTrainBatchSize"s);
    kTestBatchSize = get_option_from_map<decltype(kTestBatchSize)>("kTestBatchSize"s);
    kNumberOfEpochs = get_option_from_map<decltype(kNumberOfEpochs)>("kNumberOfEpochs"s);
    kLogInterval = get_option_from_map<decltype(kLogInterval)>("kLogInterval"s);

    validate_conf();
}

void conf::validate_conf(){
    ;
}

conf::conf(std::istream& fs){
    read_conf(fs);
    options_map = options_map_t{};
}
