#pragma once
#include <glew/GL/glew.h>
#include <SDL.h>
#include <SDL_opengl.h>
#include <string>
#include <chrono>
#include <array>

#include "settings.h"
#include "model.h"
#include "mouse_data.h"

namespace jtk
  {
  class buffer_object;
  class shader_program;
  class frame_buffer_object;
  class render_buffer;
  class texture;
  class vertex_array_object;
  }

class view
  {
  public:
    view();
    ~view(); 

    void loop();

  private:
    void _poll_for_events();
    void _imgui_ui();
    void _setup_blit_gl_objects(bool fullscreen);
    void _setup_gl_objects();
    void _log_window();
    void _control_window();
    void _script_window();
    void _destroy_gl_objects();
    void _destroy_blit_gl_objects();
    void _prepare_render();

  private:
    SDL_Window* _window;    
    uint32_t _w, _h, _viewport_w, _viewport_h, _viewport_pos_x, _viewport_pos_y;
    bool _quit;
    settings _settings;    
    jtk::frame_buffer_object* _fbo;  
    jtk::buffer_object* _vbo_array_blit;
    jtk::buffer_object* _vbo_index_blit;
    jtk::vertex_array_object* _vao_blit;
    jtk::shader_program* _program;   
    jtk::shader_program* _program_blit;
    mouse_data _md;
    model _m;
    int _lifting_scheme;
    int _function_type;
    int _test_function;
    int _space; // 0 is spline space, 1 is wavelet spline
    int _level;
    double _threshold;
    int _operation;
    int _smooth_level;
    std::string _wavelet_rules;
  };