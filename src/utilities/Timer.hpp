/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */



#ifndef UTILITIES_TIMER_MANAGER_HPP
#define UTILITIES_TIMER_MANAGER_HPP

#include <chrono>
#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <stdio.h>
#include <iomanip>
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

namespace utils
{

// simple clock. no effort currently to check for proper state. can add if needed.
struct Watch : private std::chrono::steady_clock{
  std::string name;
  time_point  start_;
  bool in_use = false;
  int ncalls = 0;
  double total_time = 0.0;
  Watch(std::string name_ = "") : name(name_), start_{now()} {}
  ~Watch() = default;
  void start() { 
    in_use = true; /* should abort if in_use=true*/ 
    start_=now();
  }
  double time() { return std::chrono::duration<double>(now() - start_).count(); }
  void stop() { 
    total_time += std::chrono::duration<double>(now() - start_).count();
    ncalls++;
    in_use = false; /* should abort if in_use=false*/ 
  } 
  double elapsed() { return total_time; }
  double average() { return (ncalls > 0 ? total_time/double(ncalls) : 0.0); }
  int number_of_calls() { return ncalls; }
  void reset() {
    total_time = 0.0;
    ncalls = 0;
    in_use = false;
  }
};

// simple lambda function to time a callable
inline auto function_timer = [] (auto f) {
  Watch timer;
  timer.start();
  f();
  timer.stop();
  return timer.elapsed();
};

// simple lambda function to time a callable
template<class... A>
auto function_timer_with_params = [] (auto f, A&&... args) {
  Watch timer;
  timer.start();
  f(std::forward<A>(args)...);
  timer.stop();
  return timer.elapsed();
};

// very simple flat timer
// template<class Watch>
class TimerManager
{
private:
  std::vector<Watch> timers;
  std::map<std::string, int> id2pos;

  int getOrAdd(std::string const& str)
  {
    std::map<std::string, int>::iterator it = id2pos.find(str);
    if (it != id2pos.end()) {
      return it->second;
    } else {
      timers.emplace_back(Watch(str));
      int n = timers.size() - 1;
      id2pos[str] = n;
      return n;
    }
  }

  int getPos(std::string const& str)
  {
    std::map<std::string, int>::iterator it = id2pos.find(str);
    if (it != id2pos.end())
      return it->second;
    else
      APP_ABORT(" Error in TimerManager: Unregistered timer " + str + "\n ");
    return 0;
  } 

  int checkPos(int n)
  {
    if(n >= timers.size())
      APP_ABORT(" Error in TimerManager n too large n:" + std::to_string(n) + "");
    return n;
  } 

public:
  TimerManager() { timers.reserve(100); }

  // removes all timers
  void clear_timers() {
    timers.clear();
    id2pos.clear();
  }

  // returns the integer associated with a timer id. If the timer does not exist, it will be added. 
  int add(const std::string& str) { return getOrAdd(str); }

  // starts a timer with name 'str'. If the timer does not exist, it will be added. 
  void start(const std::string& str) { timers[ getOrAdd(str) ].start(); }

  void start(int n) { timers[ checkPos(n) ].start(); }

  void stop(const std::string& str) { timers[ getPos(str) ].stop(); }

  void stop(int n) { timers[ checkPos(n) ].stop(); }

  double elapsed(const std::string& str) { return timers[ getPos(str) ].elapsed(); }

  double elapsed(int n){ return timers[ checkPos(n) ].elapsed(); }

  double average(const std::string& str) { return timers[ getPos(str) ].average(); }

  double average(int n) { return timers[ checkPos(n) ].average(); }

  int number_of_calls(const std::string& str) { return timers[ getPos(str) ].number_of_calls(); }

  int number_of_calls(int n) { return timers[ checkPos(n) ].number_of_calls(); }

  void reset(const std::string& str) {  timers[ getPos(str) ].reset(); }

  void reset(int n) {  timers[ checkPos(n) ].reset(); }

  void reset_all()
  {
    for (auto& t : timers) t.reset();
  }

  void reset() { reset_all(); }

  void print_elapsed(const std::string& str)
  {
    int n = getPos(str);
    app_log(1, " Elapsed time in {}: {} ", str, timers[n].elapsed());
  }

  void print_average(const std::string& str)
  {
    int n = getPos(str);
    app_log(1, " Average time in {}: {}", str, timers[n].average());
  }

  void print_average_all()
  {
    app_log(1, " Average times: ");
    for (auto& t : timers)
      app_log(1, " {}: {} ", t.name.c_str(), t.average());
    app_log(1, "");
  }

  void print_all()
  {
    app_log(1,"***************************************************************************");
    app_log(1, "{:>30}:{:>16}{:>16}{:>9}", "Timer Name", "Elapsed (s)", "Averaged (s)", 
							"# calls");
    app_log(1,"***************************************************************************");
    for (auto& t : timers) 
      app_log(1, "{:>30}:{:16.8g}{:16.8g}{:9d}", t.name.c_str(), t.elapsed(),
						   t.average(), t.number_of_calls());
    app_log(1,"***************************************************************************");
    app_log_flush();
  }
};

}

#endif //
