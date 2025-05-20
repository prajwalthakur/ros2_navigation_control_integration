#ifndef CUSTOM_DIFF_CONTROLLER__VISIBILITY_CONTROL_H_
#define CUSTOM_DIFF_CONTROLLER__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define CUSTOM_DIFF_CONTROLLER_EXPORT __attribute__ ((dllexport))
    #define CUSTOM_DIFF_CONTROLLER_IMPORT __attribute__ ((dllimport))
  #else
    #define CUSTOM_DIFF_CONTROLLER_EXPORT __declspec(dllexport)
    #define CUSTOM_DIFF_CONTROLLER_IMPORT __declspec(dllimport)
  #endif
  #ifdef CUSTOM_DIFF_CONTROLLER_BUILDING_LIBRARY
    #define CUSTOM_DIFF_CONTROLLER_PUBLIC CUSTOM_DIFF_CONTROLLER_EXPORT
  #else
    #define CUSTOM_DIFF_CONTROLLER_PUBLIC CUSTOM_DIFF_CONTROLLER_IMPORT
  #endif
  #define CUSTOM_DIFF_CONTROLLER_PUBLIC_TYPE CUSTOM_DIFF_CONTROLLER_PUBLIC
  #define CUSTOM_DIFF_CONTROLLER_LOCAL
#else
  #define CUSTOM_DIFF_CONTROLLER_EXPORT __attribute__ ((visibility("default")))
  #define CUSTOM_DIFF_CONTROLLER_IMPORT
  #if __GNUC__ >= 4
    #define CUSTOM_DIFF_CONTROLLER_PUBLIC __attribute__ ((visibility("default")))
    #define CUSTOM_DIFF_CONTROLLER_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define CUSTOM_DIFF_CONTROLLER_PUBLIC
    #define CUSTOM_DIFF_CONTROLLER_LOCAL
  #endif
  #define CUSTOM_DIFF_CONTROLLER_PUBLIC_TYPE
#endif

#endif  // CUSTOM_DIFF_CONTROLLER__VISIBILITY_CONTROL_H_
