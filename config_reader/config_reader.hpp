//
// Created by home on 20.02.2022.
// Modification of Indrekis' file
// https://github.com/indrekis/integrate1d_sample/blob/main/integrate1d_conf.hpp
//

#ifndef TEMPLATE_INTEGRATE_CONF_H
#define TEMPLATE_INTEGRATE_CONF_H

#include <string>
#include <optional>
#include <map>
#include <iosfwd>       // lightweight realisation of iostream
#include <exception>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <iostream>

#ifdef PRINT_INTERMEDIATE_STEPS
#include <iostream>
#endif

#include "string_tools.h"

// here runtime error is used as provider of
// constructor that accepts a string error msg
// which is called by what() : const char pointer is returned that points
// at a C string that has the same string as was passed into the constructor.
// https://stackoverflow.com/questions/1569726/difference-stdruntime-error-vs-stdexception
struct config_error : public std::runtime_error {
    using runtime_error::runtime_error;
};

struct wrong_option_line_error : public config_error{
    using config_error::config_error;
};

struct wrong_option_arg_error : public config_error{
    using config_error::config_error;
};

struct option_duplicated_error : public config_error{
    using config_error::config_error;
};

struct option_not_found_error : public config_error{
    using config_error::config_error;
};

struct wrong_logical_option_arg_error : public wrong_option_arg_error{
    using wrong_option_arg_error::wrong_option_arg_error;
};

class conf {
public:
    std::filesystem::path kDataRoot;
    size_t kTrainBatchSize;
    size_t kTestBatchSize;
    size_t kNumberOfEpochs;
    size_t kLogInterval;

    // one time constructor
    conf(std::istream& fs);

private:
    using parse_string_ret_t = std::pair<std::string, std::optional<std::string>>;
    using options_map_t = std::map<std::string, std::string>;

    static parse_string_ret_t parse_string(std::string arg);
    void file_to_options_map(std::istream& cf);

    template<typename T>
    T get_option_from_map(const std::string& option_name) const;

    void read_conf(std::istream& cf);
    void validate_conf();

    options_map_t options_map;
};

#ifdef PRINT_INTERMEDIATE_STEPS
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::optional<T>& arg)
{
    if(arg) os << *arg;
    return os;
}

#ifdef __GNUC__ // Clang too
template<typename T>
void print_type() { std::cout << __PRETTY_FUNCTION__ << '\n'; }
#endif
#endif

#endif //TEMPLATE_INTEGRATE_CONF_H