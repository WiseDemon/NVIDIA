# Makefile generated by XPJ for linux32
-include Makefile.custom
ProjectName = SnippetVehicle4W
SnippetVehicle4W_cppfiles   += ./../../SnippetCommon/ClassicMain.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicle4W/SnippetVehicle4W.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicle4W/SnippetVehicle4WRender.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicle4WCreate.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleCreate.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleFilterShader.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleNoDriveCreate.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleRaycast.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleTankCreate.cpp
SnippetVehicle4W_cppfiles   += ./../../SnippetVehicleCommon/SnippetVehicleTireFriction.cpp

SnippetVehicle4W_cpp_debug_dep    = $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_c_debug_dep      = $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_debug_dep      = $(SnippetVehicle4W_cpp_debug_dep) $(SnippetVehicle4W_c_debug_dep)
-include $(SnippetVehicle4W_debug_dep)
SnippetVehicle4W_cpp_checked_dep    = $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_c_checked_dep      = $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_checked_dep      = $(SnippetVehicle4W_cpp_checked_dep) $(SnippetVehicle4W_c_checked_dep)
-include $(SnippetVehicle4W_checked_dep)
SnippetVehicle4W_cpp_profile_dep    = $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_c_profile_dep      = $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_profile_dep      = $(SnippetVehicle4W_cpp_profile_dep) $(SnippetVehicle4W_c_profile_dep)
-include $(SnippetVehicle4W_profile_dep)
SnippetVehicle4W_cpp_release_dep    = $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_c_release_dep      = $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_release_dep      = $(SnippetVehicle4W_cpp_release_dep) $(SnippetVehicle4W_c_release_dep)
-include $(SnippetVehicle4W_release_dep)
SnippetVehicle4W_debug_hpaths    := 
SnippetVehicle4W_debug_hpaths    += ./../../../Include
SnippetVehicle4W_debug_lpaths    := 
SnippetVehicle4W_debug_lpaths    += ./../../../Lib/linux32
SnippetVehicle4W_debug_lpaths    += ./../../lib/linux32
SnippetVehicle4W_debug_lpaths    += ./../../../Bin/linux32
SnippetVehicle4W_debug_lpaths    += ./../../lib/linux32
SnippetVehicle4W_debug_defines   := $(SnippetVehicle4W_custom_defines)
SnippetVehicle4W_debug_defines   += PHYSX_PROFILE_SDK
SnippetVehicle4W_debug_defines   += RENDER_SNIPPET
SnippetVehicle4W_debug_defines   += _DEBUG
SnippetVehicle4W_debug_defines   += PX_DEBUG
SnippetVehicle4W_debug_defines   += PX_CHECKED
SnippetVehicle4W_debug_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicle4W_debug_libraries := 
SnippetVehicle4W_debug_libraries += SnippetRenderDEBUG
SnippetVehicle4W_debug_libraries += SnippetUtilsDEBUG
SnippetVehicle4W_debug_libraries += PhysX3DEBUG_x86
SnippetVehicle4W_debug_libraries += PhysX3CommonDEBUG_x86
SnippetVehicle4W_debug_libraries += PhysX3CookingDEBUG_x86
SnippetVehicle4W_debug_libraries += PhysX3CharacterKinematicDEBUG_x86
SnippetVehicle4W_debug_libraries += PhysX3ExtensionsDEBUG
SnippetVehicle4W_debug_libraries += PhysX3VehicleDEBUG
SnippetVehicle4W_debug_libraries += PhysXProfileSDKDEBUG
SnippetVehicle4W_debug_libraries += PhysXVisualDebuggerSDKDEBUG
SnippetVehicle4W_debug_libraries += PxTaskDEBUG
SnippetVehicle4W_debug_libraries += SnippetUtilsDEBUG
SnippetVehicle4W_debug_libraries += SnippetRenderDEBUG
SnippetVehicle4W_debug_libraries += GL
SnippetVehicle4W_debug_libraries += GLU
SnippetVehicle4W_debug_libraries += glut
SnippetVehicle4W_debug_libraries += X11
SnippetVehicle4W_debug_libraries += rt
SnippetVehicle4W_debug_libraries += pthread
SnippetVehicle4W_debug_common_cflags	:= $(SnippetVehicle4W_custom_cflags)
SnippetVehicle4W_debug_common_cflags    += -MMD
SnippetVehicle4W_debug_common_cflags    += $(addprefix -D, $(SnippetVehicle4W_debug_defines))
SnippetVehicle4W_debug_common_cflags    += $(addprefix -I, $(SnippetVehicle4W_debug_hpaths))
SnippetVehicle4W_debug_common_cflags  += -m32
SnippetVehicle4W_debug_cflags	:= $(SnippetVehicle4W_debug_common_cflags)
SnippetVehicle4W_debug_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_debug_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_debug_cflags  += -Wno-long-long
SnippetVehicle4W_debug_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_debug_cflags  += -Wno-unused-parameter
SnippetVehicle4W_debug_cflags  += -g3 -gdwarf-2
SnippetVehicle4W_debug_cppflags	:= $(SnippetVehicle4W_debug_common_cflags)
SnippetVehicle4W_debug_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_debug_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_debug_cppflags  += -Wno-long-long
SnippetVehicle4W_debug_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_debug_cppflags  += -Wno-unused-parameter
SnippetVehicle4W_debug_cppflags  += -g3 -gdwarf-2
SnippetVehicle4W_debug_lflags    := $(SnippetVehicle4W_custom_lflags)
SnippetVehicle4W_debug_lflags    += $(addprefix -L, $(SnippetVehicle4W_debug_lpaths))
SnippetVehicle4W_debug_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicle4W_debug_libraries)) -Wl,--end-group
SnippetVehicle4W_debug_lflags  += -lrt
SnippetVehicle4W_debug_lflags  += -Wl,-rpath ./
SnippetVehicle4W_debug_lflags  += -m32
SnippetVehicle4W_debug_objsdir  = $(OBJS_DIR)/SnippetVehicle4W_debug
SnippetVehicle4W_debug_cpp_o    = $(addprefix $(SnippetVehicle4W_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_debug_c_o      = $(addprefix $(SnippetVehicle4W_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_debug_obj      = $(SnippetVehicle4W_debug_cpp_o) $(SnippetVehicle4W_debug_c_o)
SnippetVehicle4W_debug_bin      := ./../../../Bin/linux32/SnippetVehicle4WDEBUG

clean_SnippetVehicle4W_debug: 
	@$(ECHO) clean SnippetVehicle4W debug
	@$(RMDIR) $(SnippetVehicle4W_debug_objsdir)
	@$(RMDIR) $(SnippetVehicle4W_debug_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicle4W/debug

build_SnippetVehicle4W_debug: postbuild_SnippetVehicle4W_debug
postbuild_SnippetVehicle4W_debug: mainbuild_SnippetVehicle4W_debug
mainbuild_SnippetVehicle4W_debug: prebuild_SnippetVehicle4W_debug $(SnippetVehicle4W_debug_bin)
prebuild_SnippetVehicle4W_debug:

$(SnippetVehicle4W_debug_bin): $(SnippetVehicle4W_debug_obj) build_SnippetRender_debug build_SnippetUtils_debug 
	@mkdir -p `dirname ./../../../Bin/linux32/SnippetVehicle4WDEBUG`
	@$(CCLD) $(SnippetVehicle4W_debug_obj) $(SnippetVehicle4W_debug_lflags) -o $(SnippetVehicle4W_debug_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicle4W_debug_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicle4W_debug_cpp_o): $(SnippetVehicle4W_debug_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling debug $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicle4W_debug_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))))
	@cp $(SnippetVehicle4W_debug_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  rm -f $(SnippetVehicle4W_debug_DEPDIR).d

$(SnippetVehicle4W_debug_c_o): $(SnippetVehicle4W_debug_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling debug $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicle4W_debug_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))))
	@cp $(SnippetVehicle4W_debug_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_debug_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  rm -f $(SnippetVehicle4W_debug_DEPDIR).d

SnippetVehicle4W_checked_hpaths    := 
SnippetVehicle4W_checked_hpaths    += ./../../../Include
SnippetVehicle4W_checked_lpaths    := 
SnippetVehicle4W_checked_lpaths    += ./../../../Lib/linux32
SnippetVehicle4W_checked_lpaths    += ./../../lib/linux32
SnippetVehicle4W_checked_lpaths    += ./../../../Bin/linux32
SnippetVehicle4W_checked_lpaths    += ./../../lib/linux32
SnippetVehicle4W_checked_defines   := $(SnippetVehicle4W_custom_defines)
SnippetVehicle4W_checked_defines   += PHYSX_PROFILE_SDK
SnippetVehicle4W_checked_defines   += RENDER_SNIPPET
SnippetVehicle4W_checked_defines   += NDEBUG
SnippetVehicle4W_checked_defines   += PX_CHECKED
SnippetVehicle4W_checked_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicle4W_checked_libraries := 
SnippetVehicle4W_checked_libraries += SnippetRenderCHECKED
SnippetVehicle4W_checked_libraries += SnippetUtilsCHECKED
SnippetVehicle4W_checked_libraries += PhysX3CHECKED_x86
SnippetVehicle4W_checked_libraries += PhysX3CommonCHECKED_x86
SnippetVehicle4W_checked_libraries += PhysX3CookingCHECKED_x86
SnippetVehicle4W_checked_libraries += PhysX3CharacterKinematicCHECKED_x86
SnippetVehicle4W_checked_libraries += PhysX3ExtensionsCHECKED
SnippetVehicle4W_checked_libraries += PhysX3VehicleCHECKED
SnippetVehicle4W_checked_libraries += PhysXProfileSDKCHECKED
SnippetVehicle4W_checked_libraries += PhysXVisualDebuggerSDKCHECKED
SnippetVehicle4W_checked_libraries += PxTaskCHECKED
SnippetVehicle4W_checked_libraries += SnippetUtilsCHECKED
SnippetVehicle4W_checked_libraries += SnippetRenderCHECKED
SnippetVehicle4W_checked_libraries += GL
SnippetVehicle4W_checked_libraries += GLU
SnippetVehicle4W_checked_libraries += glut
SnippetVehicle4W_checked_libraries += X11
SnippetVehicle4W_checked_libraries += rt
SnippetVehicle4W_checked_libraries += pthread
SnippetVehicle4W_checked_common_cflags	:= $(SnippetVehicle4W_custom_cflags)
SnippetVehicle4W_checked_common_cflags    += -MMD
SnippetVehicle4W_checked_common_cflags    += $(addprefix -D, $(SnippetVehicle4W_checked_defines))
SnippetVehicle4W_checked_common_cflags    += $(addprefix -I, $(SnippetVehicle4W_checked_hpaths))
SnippetVehicle4W_checked_common_cflags  += -m32
SnippetVehicle4W_checked_cflags	:= $(SnippetVehicle4W_checked_common_cflags)
SnippetVehicle4W_checked_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_checked_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_checked_cflags  += -Wno-long-long
SnippetVehicle4W_checked_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_checked_cflags  += -Wno-unused-parameter
SnippetVehicle4W_checked_cflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
SnippetVehicle4W_checked_cppflags	:= $(SnippetVehicle4W_checked_common_cflags)
SnippetVehicle4W_checked_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_checked_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_checked_cppflags  += -Wno-long-long
SnippetVehicle4W_checked_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_checked_cppflags  += -Wno-unused-parameter
SnippetVehicle4W_checked_cppflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
SnippetVehicle4W_checked_lflags    := $(SnippetVehicle4W_custom_lflags)
SnippetVehicle4W_checked_lflags    += $(addprefix -L, $(SnippetVehicle4W_checked_lpaths))
SnippetVehicle4W_checked_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicle4W_checked_libraries)) -Wl,--end-group
SnippetVehicle4W_checked_lflags  += -lrt
SnippetVehicle4W_checked_lflags  += -Wl,-rpath ./
SnippetVehicle4W_checked_lflags  += -m32
SnippetVehicle4W_checked_objsdir  = $(OBJS_DIR)/SnippetVehicle4W_checked
SnippetVehicle4W_checked_cpp_o    = $(addprefix $(SnippetVehicle4W_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_checked_c_o      = $(addprefix $(SnippetVehicle4W_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_checked_obj      = $(SnippetVehicle4W_checked_cpp_o) $(SnippetVehicle4W_checked_c_o)
SnippetVehicle4W_checked_bin      := ./../../../Bin/linux32/SnippetVehicle4WCHECKED

clean_SnippetVehicle4W_checked: 
	@$(ECHO) clean SnippetVehicle4W checked
	@$(RMDIR) $(SnippetVehicle4W_checked_objsdir)
	@$(RMDIR) $(SnippetVehicle4W_checked_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicle4W/checked

build_SnippetVehicle4W_checked: postbuild_SnippetVehicle4W_checked
postbuild_SnippetVehicle4W_checked: mainbuild_SnippetVehicle4W_checked
mainbuild_SnippetVehicle4W_checked: prebuild_SnippetVehicle4W_checked $(SnippetVehicle4W_checked_bin)
prebuild_SnippetVehicle4W_checked:

$(SnippetVehicle4W_checked_bin): $(SnippetVehicle4W_checked_obj) build_SnippetRender_checked build_SnippetUtils_checked 
	@mkdir -p `dirname ./../../../Bin/linux32/SnippetVehicle4WCHECKED`
	@$(CCLD) $(SnippetVehicle4W_checked_obj) $(SnippetVehicle4W_checked_lflags) -o $(SnippetVehicle4W_checked_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicle4W_checked_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicle4W_checked_cpp_o): $(SnippetVehicle4W_checked_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling checked $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicle4W_checked_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))))
	@cp $(SnippetVehicle4W_checked_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  rm -f $(SnippetVehicle4W_checked_DEPDIR).d

$(SnippetVehicle4W_checked_c_o): $(SnippetVehicle4W_checked_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling checked $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicle4W_checked_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))))
	@cp $(SnippetVehicle4W_checked_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_checked_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  rm -f $(SnippetVehicle4W_checked_DEPDIR).d

SnippetVehicle4W_profile_hpaths    := 
SnippetVehicle4W_profile_hpaths    += ./../../../Include
SnippetVehicle4W_profile_lpaths    := 
SnippetVehicle4W_profile_lpaths    += ./../../../Lib/linux32
SnippetVehicle4W_profile_lpaths    += ./../../lib/linux32
SnippetVehicle4W_profile_lpaths    += ./../../../Bin/linux32
SnippetVehicle4W_profile_lpaths    += ./../../lib/linux32
SnippetVehicle4W_profile_defines   := $(SnippetVehicle4W_custom_defines)
SnippetVehicle4W_profile_defines   += PHYSX_PROFILE_SDK
SnippetVehicle4W_profile_defines   += RENDER_SNIPPET
SnippetVehicle4W_profile_defines   += NDEBUG
SnippetVehicle4W_profile_defines   += PX_PROFILE
SnippetVehicle4W_profile_defines   += PX_SUPPORT_VISUAL_DEBUGGER
SnippetVehicle4W_profile_libraries := 
SnippetVehicle4W_profile_libraries += SnippetRenderPROFILE
SnippetVehicle4W_profile_libraries += SnippetUtilsPROFILE
SnippetVehicle4W_profile_libraries += PhysX3PROFILE_x86
SnippetVehicle4W_profile_libraries += PhysX3CommonPROFILE_x86
SnippetVehicle4W_profile_libraries += PhysX3CookingPROFILE_x86
SnippetVehicle4W_profile_libraries += PhysX3CharacterKinematicPROFILE_x86
SnippetVehicle4W_profile_libraries += PhysX3ExtensionsPROFILE
SnippetVehicle4W_profile_libraries += PhysX3VehiclePROFILE
SnippetVehicle4W_profile_libraries += PhysXProfileSDKPROFILE
SnippetVehicle4W_profile_libraries += PhysXVisualDebuggerSDKPROFILE
SnippetVehicle4W_profile_libraries += PxTaskPROFILE
SnippetVehicle4W_profile_libraries += SnippetUtilsPROFILE
SnippetVehicle4W_profile_libraries += SnippetRenderPROFILE
SnippetVehicle4W_profile_libraries += GL
SnippetVehicle4W_profile_libraries += GLU
SnippetVehicle4W_profile_libraries += glut
SnippetVehicle4W_profile_libraries += X11
SnippetVehicle4W_profile_libraries += rt
SnippetVehicle4W_profile_libraries += pthread
SnippetVehicle4W_profile_common_cflags	:= $(SnippetVehicle4W_custom_cflags)
SnippetVehicle4W_profile_common_cflags    += -MMD
SnippetVehicle4W_profile_common_cflags    += $(addprefix -D, $(SnippetVehicle4W_profile_defines))
SnippetVehicle4W_profile_common_cflags    += $(addprefix -I, $(SnippetVehicle4W_profile_hpaths))
SnippetVehicle4W_profile_common_cflags  += -m32
SnippetVehicle4W_profile_cflags	:= $(SnippetVehicle4W_profile_common_cflags)
SnippetVehicle4W_profile_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_profile_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_profile_cflags  += -Wno-long-long
SnippetVehicle4W_profile_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_profile_cflags  += -Wno-unused-parameter
SnippetVehicle4W_profile_cflags  += -O3 -fno-strict-aliasing
SnippetVehicle4W_profile_cppflags	:= $(SnippetVehicle4W_profile_common_cflags)
SnippetVehicle4W_profile_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_profile_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_profile_cppflags  += -Wno-long-long
SnippetVehicle4W_profile_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_profile_cppflags  += -Wno-unused-parameter
SnippetVehicle4W_profile_cppflags  += -O3 -fno-strict-aliasing
SnippetVehicle4W_profile_lflags    := $(SnippetVehicle4W_custom_lflags)
SnippetVehicle4W_profile_lflags    += $(addprefix -L, $(SnippetVehicle4W_profile_lpaths))
SnippetVehicle4W_profile_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicle4W_profile_libraries)) -Wl,--end-group
SnippetVehicle4W_profile_lflags  += -lrt
SnippetVehicle4W_profile_lflags  += -Wl,-rpath ./
SnippetVehicle4W_profile_lflags  += -m32
SnippetVehicle4W_profile_objsdir  = $(OBJS_DIR)/SnippetVehicle4W_profile
SnippetVehicle4W_profile_cpp_o    = $(addprefix $(SnippetVehicle4W_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_profile_c_o      = $(addprefix $(SnippetVehicle4W_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_profile_obj      = $(SnippetVehicle4W_profile_cpp_o) $(SnippetVehicle4W_profile_c_o)
SnippetVehicle4W_profile_bin      := ./../../../Bin/linux32/SnippetVehicle4WPROFILE

clean_SnippetVehicle4W_profile: 
	@$(ECHO) clean SnippetVehicle4W profile
	@$(RMDIR) $(SnippetVehicle4W_profile_objsdir)
	@$(RMDIR) $(SnippetVehicle4W_profile_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicle4W/profile

build_SnippetVehicle4W_profile: postbuild_SnippetVehicle4W_profile
postbuild_SnippetVehicle4W_profile: mainbuild_SnippetVehicle4W_profile
mainbuild_SnippetVehicle4W_profile: prebuild_SnippetVehicle4W_profile $(SnippetVehicle4W_profile_bin)
prebuild_SnippetVehicle4W_profile:

$(SnippetVehicle4W_profile_bin): $(SnippetVehicle4W_profile_obj) build_SnippetRender_profile build_SnippetUtils_profile 
	@mkdir -p `dirname ./../../../Bin/linux32/SnippetVehicle4WPROFILE`
	@$(CCLD) $(SnippetVehicle4W_profile_obj) $(SnippetVehicle4W_profile_lflags) -o $(SnippetVehicle4W_profile_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicle4W_profile_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicle4W_profile_cpp_o): $(SnippetVehicle4W_profile_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling profile $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicle4W_profile_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))))
	@cp $(SnippetVehicle4W_profile_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  rm -f $(SnippetVehicle4W_profile_DEPDIR).d

$(SnippetVehicle4W_profile_c_o): $(SnippetVehicle4W_profile_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling profile $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicle4W_profile_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))))
	@cp $(SnippetVehicle4W_profile_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_profile_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  rm -f $(SnippetVehicle4W_profile_DEPDIR).d

SnippetVehicle4W_release_hpaths    := 
SnippetVehicle4W_release_hpaths    += ./../../../Include
SnippetVehicle4W_release_lpaths    := 
SnippetVehicle4W_release_lpaths    += ./../../../Lib/linux32
SnippetVehicle4W_release_lpaths    += ./../../lib/linux32
SnippetVehicle4W_release_lpaths    += ./../../../Bin/linux32
SnippetVehicle4W_release_lpaths    += ./../../lib/linux32
SnippetVehicle4W_release_defines   := $(SnippetVehicle4W_custom_defines)
SnippetVehicle4W_release_defines   += PHYSX_PROFILE_SDK
SnippetVehicle4W_release_defines   += RENDER_SNIPPET
SnippetVehicle4W_release_defines   += NDEBUG
SnippetVehicle4W_release_libraries := 
SnippetVehicle4W_release_libraries += SnippetRender
SnippetVehicle4W_release_libraries += SnippetUtils
SnippetVehicle4W_release_libraries += PhysX3_x86
SnippetVehicle4W_release_libraries += PhysX3Common_x86
SnippetVehicle4W_release_libraries += PhysX3Cooking_x86
SnippetVehicle4W_release_libraries += PhysX3CharacterKinematic_x86
SnippetVehicle4W_release_libraries += PhysX3Extensions
SnippetVehicle4W_release_libraries += PhysX3Vehicle
SnippetVehicle4W_release_libraries += PhysXProfileSDK
SnippetVehicle4W_release_libraries += PhysXVisualDebuggerSDK
SnippetVehicle4W_release_libraries += PxTask
SnippetVehicle4W_release_libraries += SnippetUtils
SnippetVehicle4W_release_libraries += SnippetRender
SnippetVehicle4W_release_libraries += GL
SnippetVehicle4W_release_libraries += GLU
SnippetVehicle4W_release_libraries += glut
SnippetVehicle4W_release_libraries += X11
SnippetVehicle4W_release_libraries += rt
SnippetVehicle4W_release_libraries += pthread
SnippetVehicle4W_release_common_cflags	:= $(SnippetVehicle4W_custom_cflags)
SnippetVehicle4W_release_common_cflags    += -MMD
SnippetVehicle4W_release_common_cflags    += $(addprefix -D, $(SnippetVehicle4W_release_defines))
SnippetVehicle4W_release_common_cflags    += $(addprefix -I, $(SnippetVehicle4W_release_hpaths))
SnippetVehicle4W_release_common_cflags  += -m32
SnippetVehicle4W_release_cflags	:= $(SnippetVehicle4W_release_common_cflags)
SnippetVehicle4W_release_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_release_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_release_cflags  += -Wno-long-long
SnippetVehicle4W_release_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_release_cflags  += -Wno-unused-parameter
SnippetVehicle4W_release_cflags  += -O3 -fno-strict-aliasing
SnippetVehicle4W_release_cppflags	:= $(SnippetVehicle4W_release_common_cflags)
SnippetVehicle4W_release_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
SnippetVehicle4W_release_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
SnippetVehicle4W_release_cppflags  += -Wno-long-long
SnippetVehicle4W_release_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized
SnippetVehicle4W_release_cppflags  += -Wno-unused-parameter
SnippetVehicle4W_release_cppflags  += -O3 -fno-strict-aliasing
SnippetVehicle4W_release_lflags    := $(SnippetVehicle4W_custom_lflags)
SnippetVehicle4W_release_lflags    += $(addprefix -L, $(SnippetVehicle4W_release_lpaths))
SnippetVehicle4W_release_lflags    += -Wl,--start-group $(addprefix -l, $(SnippetVehicle4W_release_libraries)) -Wl,--end-group
SnippetVehicle4W_release_lflags  += -lrt
SnippetVehicle4W_release_lflags  += -Wl,-rpath ./
SnippetVehicle4W_release_lflags  += -m32
SnippetVehicle4W_release_objsdir  = $(OBJS_DIR)/SnippetVehicle4W_release
SnippetVehicle4W_release_cpp_o    = $(addprefix $(SnippetVehicle4W_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(SnippetVehicle4W_cppfiles)))))
SnippetVehicle4W_release_c_o      = $(addprefix $(SnippetVehicle4W_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(SnippetVehicle4W_cfiles)))))
SnippetVehicle4W_release_obj      = $(SnippetVehicle4W_release_cpp_o) $(SnippetVehicle4W_release_c_o)
SnippetVehicle4W_release_bin      := ./../../../Bin/linux32/SnippetVehicle4W

clean_SnippetVehicle4W_release: 
	@$(ECHO) clean SnippetVehicle4W release
	@$(RMDIR) $(SnippetVehicle4W_release_objsdir)
	@$(RMDIR) $(SnippetVehicle4W_release_bin)
	@$(RMDIR) $(DEPSDIR)/SnippetVehicle4W/release

build_SnippetVehicle4W_release: postbuild_SnippetVehicle4W_release
postbuild_SnippetVehicle4W_release: mainbuild_SnippetVehicle4W_release
mainbuild_SnippetVehicle4W_release: prebuild_SnippetVehicle4W_release $(SnippetVehicle4W_release_bin)
prebuild_SnippetVehicle4W_release:

$(SnippetVehicle4W_release_bin): $(SnippetVehicle4W_release_obj) build_SnippetRender_release build_SnippetUtils_release 
	@mkdir -p `dirname ./../../../Bin/linux32/SnippetVehicle4W`
	@$(CCLD) $(SnippetVehicle4W_release_obj) $(SnippetVehicle4W_release_lflags) -o $(SnippetVehicle4W_release_bin) 
	@$(ECHO) building $@ complete!

SnippetVehicle4W_release_DEPDIR = $(dir $(@))/$(*F)
$(SnippetVehicle4W_release_cpp_o): $(SnippetVehicle4W_release_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling release $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(SnippetVehicle4W_release_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))))
	@cp $(SnippetVehicle4W_release_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cppfiles))))).P; \
	  rm -f $(SnippetVehicle4W_release_DEPDIR).d

$(SnippetVehicle4W_release_c_o): $(SnippetVehicle4W_release_objsdir)/%.o:
	@$(ECHO) SnippetVehicle4W: compiling release $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(SnippetVehicle4W_release_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))))
	@cp $(SnippetVehicle4W_release_DEPDIR).d $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(SnippetVehicle4W_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/SnippetVehicle4W/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(SnippetVehicle4W_release_objsdir),, $@))), $(SnippetVehicle4W_cfiles))))).P; \
	  rm -f $(SnippetVehicle4W_release_DEPDIR).d

clean_SnippetVehicle4W:  clean_SnippetVehicle4W_debug clean_SnippetVehicle4W_checked clean_SnippetVehicle4W_profile clean_SnippetVehicle4W_release
	@$(RMDIR) $(DEPSDIR)/SnippetVehicle4W
