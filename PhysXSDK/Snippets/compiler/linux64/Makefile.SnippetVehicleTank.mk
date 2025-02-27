# Makefile generated by XPJ for linux64
-include Makefile.custom
ProjectName = SnippetVehicleTank
SnippetVehicleTank_cppfiles   += ./../../SnippetCommon/ClassicMain.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleTank/SnippetVehicleTank.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleTank/SnippetVehicleTankRender.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicle4WCreate.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleCreate.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleFilterShader.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleNoDriveCreate.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleRaycast.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleTankCreate.cpp
SnippetVehicleTank_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleTireFriction.cpp

SnippetVehicleTank_cpp_debug_dep    = $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_c_debug_dep      = $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_debug_dep      = $(SnippetVehicleTank_cpp_debug_dep) $(SnippetVehicleTank_c_debug_dep)
-include $(SnippetVehicleTank_debug_dep)
SnippetVehicleTank_cpp_checked_dep    = $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_c_checked_dep      = $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_checked_dep      = $(SnippetVehicleTank_cpp_checked_dep) $(SnippetVehicleTank_c_checked_dep)
-include $(SnippetVehicleTank_checked_dep)
SnippetVehicleTank_cpp_profile_dep    = $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_c_profile_dep      = $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_profile_dep      = $(SnippetVehicleTank_cpp_profile_dep) $(SnippetVehicleTank_c_profile_dep)
-include $(SnippetVehicleTank_profile_dep)
SnippetVehicleTank_cpp_release_dep    = $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_c_release_dep      = $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_release_dep      = $(SnippetVehicleTank_cpp_release_dep) $(SnippetVehicleTank_c_release_dep)
-include $(SnippetVehicleTank_release_dep)
SnippetVehicleTank_debug_hpaths    := 
SnippetVehicleTank_debug_hpaths    += ./../../../Include
SnippetVehicleTank_debug_lpaths    := 
SnippetVehicleTank_debug_lpaths    += ./../../../Lib/linux64
SnippetVehicleTank_debug_lpaths    += ./../../lib/linux64
SnippetVehicleTank_debug_lpaths    += ./../../../Bin/linux64
SnippetVehicleTank_debug_lpaths    += ./../../lib/linux64
SnippetVehicleTank_debug_defines   := $(SnippetVehicleTank_custom_defines)
SnippetVehicleTank_debug_defines   += PHYSX_PROFILE_SDK
SnippetVehicleTank_debug_defines   += RENDER_SNIPPET
SnippetVehicleTank_debug_defines   += _DEBUG
SnippetVehicleTank_debug_defines   += PX_DEBUG
SnippetVehicleTank_debug_defines   += PX_CHECKED
SnippetVehicleTank_debug_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicleTank_debug_libraries := 
SnippetVehicleTank_debug_libraries += SnippetRenderDEBUG
SnippetVehicleTank_debug_libraries += SnippetUtilsDEBUG
SnippetVehicleTank_debug_libraries += PhysX3DEBUG_x64
SnippetVehicleTank_debug_libraries += PhysX3CommonDEBUG_x64
SnippetVehicleTank_debug_libraries += PhysX3CookingDEBUG_x64
SnippetVehicleTank_debug_libraries += PhysX3CharacterKinematicDEBUG_x64
SnippetVehicleTank_debug_libraries += PhysX3ExtensionsDEBUG
SnippetVehicleTank_debug_libraries += PhysX3VehicleDEBUG
SnippetVehicleTank_debug_libraries += PhysXProfileSDKDEBUG
SnippetVehicleTank_debug_libraries += PhysXVisualDebuggerSDKDEBUG
SnippetVehicleTank_debug_libraries += PxTaskDEBUG
SnippetVehicleTank_debug_libraries += SnippetUtilsDEBUG
SnippetVehicleTank_debug_libraries += SnippetRenderDEBUG
SnippetVehicleTank_debug_libraries += GL
SnippetVehicleTank_debug_libraries += GLU
SnippetVehicleTank_debug_libraries += glut
SnippetVehicleTank_debug_libraries += X11
SnippetVehicleTank_debug_libraries += rt
SnippetVehicleTank_debug_libraries += pthread
SnippetVehicleTank_debug_common_cflags	:= $(SnippetVehicleTank_custom_cflags)
SnippetVehicleTank_debug_common_cflags    += -MMD
SnippetVehicleTank_debug_common_cflags    += $(addprefix -D, $(SnippetVehicleTank_debug_defines))
SnippetVehicleTank_debug_common_cflags    += $(addprefix -I, $(SnippetVehicleTank_debug_hpaths))
SnippetVehicleTank_debug_common_cflags  += -m64
SnippetVehicleTank_debug_cflags	:= $(SnippetVehicleTank_debug_common_cflags)
SnippetVehicleTank_debug_cflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_debug_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_debug_cflags  += -Wno-long-long
SnippetVehicleTank_debug_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_debug_cflags  += -Wno-unused-parameter
SnippetVehicleTank_debug_cflags  += -g3 -gdwarf-2
SnippetVehicleTank_debug_cppflags	:= $(SnippetVehicleTank_debug_common_cflags)
SnippetVehicleTank_debug_cppflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_debug_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_debug_cppflags  += -Wno-long-long
SnippetVehicleTank_debug_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_debug_cppflags  += -Wno-unused-parameter
SnippetVehicleTank_debug_cppflags  += -g3 -gdwarf-2
SnippetVehicleTank_debug_lflags    := $(SnippetVehicleTank_custom_lflags)
SnippetVehicleTank_debug_lflags    += $(addprefix -L, $(SnippetVehicleTank_debug_lpaths))
SnippetVehicleTank_debug_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicleTank_debug_libraries)) -Wl,--end-group
SnippetVehicleTank_debug_lflags  += -lrt
SnippetVehicleTank_debug_lflags  += -Wl,-rpath ./
SnippetVehicleTank_debug_lflags  += -m64
SnippetVehicleTank_debug_objsdir  = $(OBJS_DIR)/SnippetVehicleTank_debug
SnippetVehicleTank_debug_cpp_o    = $(addprefix $(SnippetVehicleTank_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_debug_c_o      = $(addprefix $(SnippetVehicleTank_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_debug_obj      = $(SnippetVehicleTank_debug_cpp_o) $(SnippetVehicleTank_debug_c_o)
SnippetVehicleTank_debug_bin      := ./../../../Bin/linux64/SnippetVehicleTankDEBUG

clean_SnippetVehicleTank_debug: 
	@$(ECHO) clean SnippetVehicleTank debug
	@$(RMDIR) $(SnippetVehicleTank_debug_objsdir)
	@$(RMDIR) $(SnippetVehicleTank_debug_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicleTank/debug

build_SnippetVehicleTank_debug: postbuild_SnippetVehicleTank_debug
postbuild_SnippetVehicleTank_debug: mainbuild_SnippetVehicleTank_debug
mainbuild_SnippetVehicleTank_debug: prebuild_SnippetVehicleTank_debug $(SnippetVehicleTank_debug_bin)
prebuild_SnippetVehicleTank_debug:

$(SnippetVehicleTank_debug_bin): $(SnippetVehicleTank_debug_obj) build_SnippetRender_debug build_SnippetUtils_debug 
	@mkdir -p `dirname ./../../../Bin/linux64/SnippetVehicleTankDEBUG`
	@$(CCLD) $(SnippetVehicleTank_debug_obj) $(SnippetVehicleTank_debug_lflags) -o $(SnippetVehicleTank_debug_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicleTank_debug_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicleTank_debug_cpp_o): $(SnippetVehicleTank_debug_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling debug $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicleTank_debug_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))))
	@cp $(SnippetVehicleTank_debug_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  rm -f $(SnippetVehicleTank_debug_DEPDIR).d

$(SnippetVehicleTank_debug_c_o): $(SnippetVehicleTank_debug_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling debug $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicleTank_debug_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))))
	@cp $(SnippetVehicleTank_debug_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_debug_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  rm -f $(SnippetVehicleTank_debug_DEPDIR).d

SnippetVehicleTank_checked_hpaths    := 
SnippetVehicleTank_checked_hpaths    += ./../../../Include
SnippetVehicleTank_checked_lpaths    := 
SnippetVehicleTank_checked_lpaths    += ./../../../Lib/linux64
SnippetVehicleTank_checked_lpaths    += ./../../lib/linux64
SnippetVehicleTank_checked_lpaths    += ./../../../Bin/linux64
SnippetVehicleTank_checked_lpaths    += ./../../lib/linux64
SnippetVehicleTank_checked_defines   := $(SnippetVehicleTank_custom_defines)
SnippetVehicleTank_checked_defines   += PHYSX_PROFILE_SDK
SnippetVehicleTank_checked_defines   += RENDER_SNIPPET
SnippetVehicleTank_checked_defines   += NDEBUG
SnippetVehicleTank_checked_defines   += PX_CHECKED
SnippetVehicleTank_checked_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicleTank_checked_libraries := 
SnippetVehicleTank_checked_libraries += SnippetRenderCHECKED
SnippetVehicleTank_checked_libraries += SnippetUtilsCHECKED
SnippetVehicleTank_checked_libraries += PhysX3CHECKED_x64
SnippetVehicleTank_checked_libraries += PhysX3CommonCHECKED_x64
SnippetVehicleTank_checked_libraries += PhysX3CookingCHECKED_x64
SnippetVehicleTank_checked_libraries += PhysX3CharacterKinematicCHECKED_x64
SnippetVehicleTank_checked_libraries += PhysX3ExtensionsCHECKED
SnippetVehicleTank_checked_libraries += PhysX3VehicleCHECKED
SnippetVehicleTank_checked_libraries += PhysXProfileSDKCHECKED
SnippetVehicleTank_checked_libraries += PhysXVisualDebuggerSDKCHECKED
SnippetVehicleTank_checked_libraries += PxTaskCHECKED
SnippetVehicleTank_checked_libraries += SnippetUtilsCHECKED
SnippetVehicleTank_checked_libraries += SnippetRenderCHECKED
SnippetVehicleTank_checked_libraries += GL
SnippetVehicleTank_checked_libraries += GLU
SnippetVehicleTank_checked_libraries += glut
SnippetVehicleTank_checked_libraries += X11
SnippetVehicleTank_checked_libraries += rt
SnippetVehicleTank_checked_libraries += pthread
SnippetVehicleTank_checked_common_cflags	:= $(SnippetVehicleTank_custom_cflags)
SnippetVehicleTank_checked_common_cflags    += -MMD
SnippetVehicleTank_checked_common_cflags    += $(addprefix -D, $(SnippetVehicleTank_checked_defines))
SnippetVehicleTank_checked_common_cflags    += $(addprefix -I, $(SnippetVehicleTank_checked_hpaths))
SnippetVehicleTank_checked_common_cflags  += -m64
SnippetVehicleTank_checked_cflags	:= $(SnippetVehicleTank_checked_common_cflags)
SnippetVehicleTank_checked_cflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_checked_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_checked_cflags  += -Wno-long-long
SnippetVehicleTank_checked_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_checked_cflags  += -Wno-unused-parameter
SnippetVehicleTank_checked_cflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
SnippetVehicleTank_checked_cppflags	:= $(SnippetVehicleTank_checked_common_cflags)
SnippetVehicleTank_checked_cppflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_checked_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_checked_cppflags  += -Wno-long-long
SnippetVehicleTank_checked_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_checked_cppflags  += -Wno-unused-parameter
SnippetVehicleTank_checked_cppflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
SnippetVehicleTank_checked_lflags    := $(SnippetVehicleTank_custom_lflags)
SnippetVehicleTank_checked_lflags    += $(addprefix -L, $(SnippetVehicleTank_checked_lpaths))
SnippetVehicleTank_checked_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicleTank_checked_libraries)) -Wl,--end-group
SnippetVehicleTank_checked_lflags  += -lrt
SnippetVehicleTank_checked_lflags  += -Wl,-rpath ./
SnippetVehicleTank_checked_lflags  += -m64
SnippetVehicleTank_checked_objsdir  = $(OBJS_DIR)/SnippetVehicleTank_checked
SnippetVehicleTank_checked_cpp_o    = $(addprefix $(SnippetVehicleTank_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_checked_c_o      = $(addprefix $(SnippetVehicleTank_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_checked_obj      = $(SnippetVehicleTank_checked_cpp_o) $(SnippetVehicleTank_checked_c_o)
SnippetVehicleTank_checked_bin      := ./../../../Bin/linux64/SnippetVehicleTankCHECKED

clean_SnippetVehicleTank_checked: 
	@$(ECHO) clean SnippetVehicleTank checked
	@$(RMDIR) $(SnippetVehicleTank_checked_objsdir)
	@$(RMDIR) $(SnippetVehicleTank_checked_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicleTank/checked

build_SnippetVehicleTank_checked: postbuild_SnippetVehicleTank_checked
postbuild_SnippetVehicleTank_checked: mainbuild_SnippetVehicleTank_checked
mainbuild_SnippetVehicleTank_checked: prebuild_SnippetVehicleTank_checked $(SnippetVehicleTank_checked_bin)
prebuild_SnippetVehicleTank_checked:

$(SnippetVehicleTank_checked_bin): $(SnippetVehicleTank_checked_obj) build_SnippetRender_checked build_SnippetUtils_checked 
	@mkdir -p `dirname ./../../../Bin/linux64/SnippetVehicleTankCHECKED`
	@$(CCLD) $(SnippetVehicleTank_checked_obj) $(SnippetVehicleTank_checked_lflags) -o $(SnippetVehicleTank_checked_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicleTank_checked_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicleTank_checked_cpp_o): $(SnippetVehicleTank_checked_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling checked $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicleTank_checked_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))))
	@cp $(SnippetVehicleTank_checked_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  rm -f $(SnippetVehicleTank_checked_DEPDIR).d

$(SnippetVehicleTank_checked_c_o): $(SnippetVehicleTank_checked_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling checked $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicleTank_checked_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))))
	@cp $(SnippetVehicleTank_checked_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_checked_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  rm -f $(SnippetVehicleTank_checked_DEPDIR).d

SnippetVehicleTank_profile_hpaths    := 
SnippetVehicleTank_profile_hpaths    += ./../../../Include
SnippetVehicleTank_profile_lpaths    := 
SnippetVehicleTank_profile_lpaths    += ./../../../Lib/linux64
SnippetVehicleTank_profile_lpaths    += ./../../lib/linux64
SnippetVehicleTank_profile_lpaths    += ./../../../Bin/linux64
SnippetVehicleTank_profile_lpaths    += ./../../lib/linux64
SnippetVehicleTank_profile_defines   := $(SnippetVehicleTank_custom_defines)
SnippetVehicleTank_profile_defines   += PHYSX_PROFILE_SDK
SnippetVehicleTank_profile_defines   += RENDER_SNIPPET
SnippetVehicleTank_profile_defines   += NDEBUG
SnippetVehicleTank_profile_defines   += PX_PROFILE
SnippetVehicleTank_profile_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicleTank_profile_libraries := 
SnippetVehicleTank_profile_libraries += SnippetRenderPROFILE
SnippetVehicleTank_profile_libraries += SnippetUtilsPROFILE
SnippetVehicleTank_profile_libraries += PhysX3PROFILE_x64
SnippetVehicleTank_profile_libraries += PhysX3CommonPROFILE_x64
SnippetVehicleTank_profile_libraries += PhysX3CookingPROFILE_x64
SnippetVehicleTank_profile_libraries += PhysX3CharacterKinematicPROFILE_x64
SnippetVehicleTank_profile_libraries += PhysX3ExtensionsPROFILE
SnippetVehicleTank_profile_libraries += PhysX3VehiclePROFILE
SnippetVehicleTank_profile_libraries += PhysXProfileSDKPROFILE
SnippetVehicleTank_profile_libraries += PhysXVisualDebuggerSDKPROFILE
SnippetVehicleTank_profile_libraries += PxTaskPROFILE
SnippetVehicleTank_profile_libraries += SnippetUtilsPROFILE
SnippetVehicleTank_profile_libraries += SnippetRenderPROFILE
SnippetVehicleTank_profile_libraries += GL
SnippetVehicleTank_profile_libraries += GLU
SnippetVehicleTank_profile_libraries += glut
SnippetVehicleTank_profile_libraries += X11
SnippetVehicleTank_profile_libraries += rt
SnippetVehicleTank_profile_libraries += pthread
SnippetVehicleTank_profile_common_cflags	:= $(SnippetVehicleTank_custom_cflags)
SnippetVehicleTank_profile_common_cflags    += -MMD
SnippetVehicleTank_profile_common_cflags    += $(addprefix -D, $(SnippetVehicleTank_profile_defines))
SnippetVehicleTank_profile_common_cflags    += $(addprefix -I, $(SnippetVehicleTank_profile_hpaths))
SnippetVehicleTank_profile_common_cflags  += -m64
SnippetVehicleTank_profile_cflags	:= $(SnippetVehicleTank_profile_common_cflags)
SnippetVehicleTank_profile_cflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_profile_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_profile_cflags  += -Wno-long-long
SnippetVehicleTank_profile_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_profile_cflags  += -Wno-unused-parameter
SnippetVehicleTank_profile_cflags  += -O3 -fno-strict-aliasing
SnippetVehicleTank_profile_cppflags	:= $(SnippetVehicleTank_profile_common_cflags)
SnippetVehicleTank_profile_cppflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_profile_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_profile_cppflags  += -Wno-long-long
SnippetVehicleTank_profile_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_profile_cppflags  += -Wno-unused-parameter
SnippetVehicleTank_profile_cppflags  += -O3 -fno-strict-aliasing
SnippetVehicleTank_profile_lflags    := $(SnippetVehicleTank_custom_lflags)
SnippetVehicleTank_profile_lflags    += $(addprefix -L, $(SnippetVehicleTank_profile_lpaths))
SnippetVehicleTank_profile_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicleTank_profile_libraries)) -Wl,--end-group
SnippetVehicleTank_profile_lflags  += -lrt
SnippetVehicleTank_profile_lflags  += -Wl,-rpath ./
SnippetVehicleTank_profile_lflags  += -m64
SnippetVehicleTank_profile_objsdir  = $(OBJS_DIR)/SnippetVehicleTank_profile
SnippetVehicleTank_profile_cpp_o    = $(addprefix $(SnippetVehicleTank_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_profile_c_o      = $(addprefix $(SnippetVehicleTank_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_profile_obj      = $(SnippetVehicleTank_profile_cpp_o) $(SnippetVehicleTank_profile_c_o)
SnippetVehicleTank_profile_bin      := ./../../../Bin/linux64/SnippetVehicleTankPROFILE

clean_SnippetVehicleTank_profile: 
	@$(ECHO) clean SnippetVehicleTank profile
	@$(RMDIR) $(SnippetVehicleTank_profile_objsdir)
	@$(RMDIR) $(SnippetVehicleTank_profile_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicleTank/profile

build_SnippetVehicleTank_profile: postbuild_SnippetVehicleTank_profile
postbuild_SnippetVehicleTank_profile: mainbuild_SnippetVehicleTank_profile
mainbuild_SnippetVehicleTank_profile: prebuild_SnippetVehicleTank_profile $(SnippetVehicleTank_profile_bin)
prebuild_SnippetVehicleTank_profile:

$(SnippetVehicleTank_profile_bin): $(SnippetVehicleTank_profile_obj) build_SnippetRender_profile build_SnippetUtils_profile 
	@mkdir -p `dirname ./../../../Bin/linux64/SnippetVehicleTankPROFILE`
	@$(CCLD) $(SnippetVehicleTank_profile_obj) $(SnippetVehicleTank_profile_lflags) -o $(SnippetVehicleTank_profile_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicleTank_profile_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicleTank_profile_cpp_o): $(SnippetVehicleTank_profile_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling profile $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicleTank_profile_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))))
	@cp $(SnippetVehicleTank_profile_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  rm -f $(SnippetVehicleTank_profile_DEPDIR).d

$(SnippetVehicleTank_profile_c_o): $(SnippetVehicleTank_profile_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling profile $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicleTank_profile_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))))
	@cp $(SnippetVehicleTank_profile_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_profile_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  rm -f $(SnippetVehicleTank_profile_DEPDIR).d

SnippetVehicleTank_release_hpaths    := 
SnippetVehicleTank_release_hpaths    += ./../../../Include
SnippetVehicleTank_release_lpaths    := 
SnippetVehicleTank_release_lpaths    += ./../../../Lib/linux64
SnippetVehicleTank_release_lpaths    += ./../../lib/linux64
SnippetVehicleTank_release_lpaths    += ./../../../Bin/linux64
SnippetVehicleTank_release_lpaths    += ./../../lib/linux64
SnippetVehicleTank_release_defines   := $(SnippetVehicleTank_custom_defines)
SnippetVehicleTank_release_defines   += PHYSX_PROFILE_SDK
SnippetVehicleTank_release_defines   += RENDER_SNIPPET
SnippetVehicleTank_release_defines   += NDEBUG
SnippetVehicleTank_release_libraries := 
SnippetVehicleTank_release_libraries += SnippetRender
SnippetVehicleTank_release_libraries += SnippetUtils
SnippetVehicleTank_release_libraries += PhysX3_x64
SnippetVehicleTank_release_libraries += PhysX3Common_x64
SnippetVehicleTank_release_libraries += PhysX3Cooking_x64
SnippetVehicleTank_release_libraries += PhysX3CharacterKinematic_x64
SnippetVehicleTank_release_libraries += PhysX3Extensions
SnippetVehicleTank_release_libraries += PhysX3Vehicle
SnippetVehicleTank_release_libraries += PhysXProfileSDK
SnippetVehicleTank_release_libraries += PhysXVisualDebuggerSDK
SnippetVehicleTank_release_libraries += PxTask
SnippetVehicleTank_release_libraries += SnippetUtils
SnippetVehicleTank_release_libraries += SnippetRender
SnippetVehicleTank_release_libraries += GL
SnippetVehicleTank_release_libraries += GLU
SnippetVehicleTank_release_libraries += glut
SnippetVehicleTank_release_libraries += X11
SnippetVehicleTank_release_libraries += rt
SnippetVehicleTank_release_libraries += pthread
SnippetVehicleTank_release_common_cflags	:= $(SnippetVehicleTank_custom_cflags)
SnippetVehicleTank_release_common_cflags    += -MMD
SnippetVehicleTank_release_common_cflags    += $(addprefix -D, $(SnippetVehicleTank_release_defines))
SnippetVehicleTank_release_common_cflags    += $(addprefix -I, $(SnippetVehicleTank_release_hpaths))
SnippetVehicleTank_release_common_cflags  += -m64
SnippetVehicleTank_release_cflags	:= $(SnippetVehicleTank_release_common_cflags)
SnippetVehicleTank_release_cflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_release_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_release_cflags  += -Wno-long-long
SnippetVehicleTank_release_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_release_cflags  += -Wno-unused-parameter
SnippetVehicleTank_release_cflags  += -O3 -fno-strict-aliasing
SnippetVehicleTank_release_cppflags	:= $(SnippetVehicleTank_release_common_cflags)
SnippetVehicleTank_release_cppflags  += -Werror -m64 -fPIC -msse2 -mfpmath=sse -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicleTank_release_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicleTank_release_cppflags  += -Wno-long-long
SnippetVehicleTank_release_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicleTank_release_cppflags  += -Wno-unused-parameter
SnippetVehicleTank_release_cppflags  += -O3 -fno-strict-aliasing
SnippetVehicleTank_release_lflags    := $(SnippetVehicleTank_custom_lflags)
SnippetVehicleTank_release_lflags    += $(addprefix -L, $(SnippetVehicleTank_release_lpaths))
SnippetVehicleTank_release_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicleTank_release_libraries)) -Wl,--end-group
SnippetVehicleTank_release_lflags  += -lrt
SnippetVehicleTank_release_lflags  += -Wl,-rpath ./
SnippetVehicleTank_release_lflags  += -m64
SnippetVehicleTank_release_objsdir  = $(OBJS_DIR)/SnippetVehicleTank_release
SnippetVehicleTank_release_cpp_o    = $(addprefix $(SnippetVehicleTank_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicleTank_cppfiles)))))
SnippetVehicleTank_release_c_o      = $(addprefix $(SnippetVehicleTank_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicleTank_cfiles)))))
SnippetVehicleTank_release_obj      = $(SnippetVehicleTank_release_cpp_o) $(SnippetVehicleTank_release_c_o)
SnippetVehicleTank_release_bin      := ./../../../Bin/linux64/SnippetVehicleTank

clean_SnippetVehicleTank_release: 
	@$(ECHO) clean SnippetVehicleTank release
	@$(RMDIR) $(SnippetVehicleTank_release_objsdir)
	@$(RMDIR) $(SnippetVehicleTank_release_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicleTank/release

build_SnippetVehicleTank_release: postbuild_SnippetVehicleTank_release
postbuild_SnippetVehicleTank_release: mainbuild_SnippetVehicleTank_release
mainbuild_SnippetVehicleTank_release: prebuild_SnippetVehicleTank_release $(SnippetVehicleTank_release_bin)
prebuild_SnippetVehicleTank_release:

$(SnippetVehicleTank_release_bin): $(SnippetVehicleTank_release_obj) build_SnippetRender_release build_SnippetUtils_release 
	@mkdir -p `dirname ./../../../Bin/linux64/SnippetVehicleTank`
	@$(CCLD) $(SnippetVehicleTank_release_obj) $(SnippetVehicleTank_release_lflags) -o $(SnippetVehicleTank_release_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicleTank_release_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicleTank_release_cpp_o): $(SnippetVehicleTank_release_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling release $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicleTank_release_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))))
	@cp $(SnippetVehicleTank_release_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cppfiles))))).P; \
	  rm -f $(SnippetVehicleTank_release_DEPDIR).d

$(SnippetVehicleTank_release_c_o): $(SnippetVehicleTank_release_objsdir)/%.o:
	@$(ECHO) SnippetVehicleTank: compiling release $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicleTank_release_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))))
	@cp $(SnippetVehicleTank_release_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicleTank_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicleTank/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicleTank_release_objsdir),, $@))), $(SnippetVehicleTank_cfiles))))).P; \
	  rm -f $(SnippetVehicleTank_release_DEPDIR).d

clean_SnippetVehicleTank:  clean_SnippetVehicleTank_debug clean_SnippetVehicleTank_checked clean_SnippetVehicleTank_profile clean_SnippetVehicleTank_release
	@$(RMDIR) $(DEPSDIR)/SnippetVehicleTank
