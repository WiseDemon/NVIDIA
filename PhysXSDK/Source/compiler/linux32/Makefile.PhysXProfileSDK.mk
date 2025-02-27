# Makefile generated by XPJ for linux32
-include Makefile.custom
ProjectName = PhysXProfileSDK
PhysXProfileSDK_cppfiles   += ./../../PhysXProfileSDK/PxProfileEventImpl.cpp

PhysXProfileSDK_cpp_debug_dep    = $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_c_debug_dep      = $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_debug_dep      = $(PhysXProfileSDK_cpp_debug_dep) $(PhysXProfileSDK_c_debug_dep)
-include $(PhysXProfileSDK_debug_dep)
PhysXProfileSDK_cpp_checked_dep    = $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_c_checked_dep      = $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_checked_dep      = $(PhysXProfileSDK_cpp_checked_dep) $(PhysXProfileSDK_c_checked_dep)
-include $(PhysXProfileSDK_checked_dep)
PhysXProfileSDK_cpp_profile_dep    = $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_c_profile_dep      = $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_profile_dep      = $(PhysXProfileSDK_cpp_profile_dep) $(PhysXProfileSDK_c_profile_dep)
-include $(PhysXProfileSDK_profile_dep)
PhysXProfileSDK_cpp_release_dep    = $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_c_release_dep      = $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_release_dep      = $(PhysXProfileSDK_cpp_release_dep) $(PhysXProfileSDK_c_release_dep)
-include $(PhysXProfileSDK_release_dep)
PhysXProfileSDK_debug_hpaths    := 
PhysXProfileSDK_debug_hpaths    += ./../../../Include/foundation
PhysXProfileSDK_debug_hpaths    += ./../../foundation/include
PhysXProfileSDK_debug_hpaths    += ./../../../Include/physxprofilesdk
PhysXProfileSDK_debug_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXProfileSDK_debug_hpaths    += ./../../../Include
PhysXProfileSDK_debug_lpaths    := 
PhysXProfileSDK_debug_defines   := $(PhysXProfileSDK_custom_defines)
PhysXProfileSDK_debug_defines   += PX_PHYSX_STATIC_LIB
PhysXProfileSDK_debug_defines   += _DEBUG
PhysXProfileSDK_debug_defines   += PX_DEBUG
PhysXProfileSDK_debug_defines   += PX_CHECKED
PhysXProfileSDK_debug_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXProfileSDK_debug_libraries := 
PhysXProfileSDK_debug_common_cflags	:= $(PhysXProfileSDK_custom_cflags)
PhysXProfileSDK_debug_common_cflags    += -MMD
PhysXProfileSDK_debug_common_cflags    += $(addprefix -D, $(PhysXProfileSDK_debug_defines))
PhysXProfileSDK_debug_common_cflags    += $(addprefix -I, $(PhysXProfileSDK_debug_hpaths))
PhysXProfileSDK_debug_common_cflags  += -m32
PhysXProfileSDK_debug_cflags	:= $(PhysXProfileSDK_debug_common_cflags)
PhysXProfileSDK_debug_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_debug_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_debug_cflags  += -Wno-long-long
PhysXProfileSDK_debug_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_debug_cflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_debug_cflags  += -g3 -gdwarf-2
PhysXProfileSDK_debug_cppflags	:= $(PhysXProfileSDK_debug_common_cflags)
PhysXProfileSDK_debug_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_debug_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_debug_cppflags  += -Wno-long-long
PhysXProfileSDK_debug_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_debug_cppflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_debug_cppflags  += -g3 -gdwarf-2
PhysXProfileSDK_debug_lflags    := $(PhysXProfileSDK_custom_lflags)
PhysXProfileSDK_debug_lflags    += $(addprefix -L, $(PhysXProfileSDK_debug_lpaths))
PhysXProfileSDK_debug_lflags    += -Wl,--start-group $(addprefix -l, $(PhysXProfileSDK_debug_libraries)) -Wl,--end-group
PhysXProfileSDK_debug_lflags  += -lrt
PhysXProfileSDK_debug_lflags  += -m32
PhysXProfileSDK_debug_objsdir  = $(OBJS_DIR)/PhysXProfileSDK_debug
PhysXProfileSDK_debug_cpp_o    = $(addprefix $(PhysXProfileSDK_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_debug_c_o      = $(addprefix $(PhysXProfileSDK_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_debug_obj      = $(PhysXProfileSDK_debug_cpp_o) $(PhysXProfileSDK_debug_c_o)
PhysXProfileSDK_debug_bin      := ./../../../Lib/linux32/libPhysXProfileSDKDEBUG.a

clean_PhysXProfileSDK_debug: 
	@$(ECHO) clean PhysXProfileSDK debug
	@$(RMDIR) $(PhysXProfileSDK_debug_objsdir)
	@$(RMDIR) $(PhysXProfileSDK_debug_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXProfileSDK/debug

build_PhysXProfileSDK_debug: postbuild_PhysXProfileSDK_debug
postbuild_PhysXProfileSDK_debug: mainbuild_PhysXProfileSDK_debug
mainbuild_PhysXProfileSDK_debug: prebuild_PhysXProfileSDK_debug $(PhysXProfileSDK_debug_bin)
prebuild_PhysXProfileSDK_debug:

$(PhysXProfileSDK_debug_bin): $(PhysXProfileSDK_debug_obj) 
	@mkdir -p `dirname ./../../../Lib/linux32/libPhysXProfileSDKDEBUG.a`
	@$(AR) rcs $(PhysXProfileSDK_debug_bin) $(PhysXProfileSDK_debug_obj)
	@$(ECHO) building $@ complete!

PhysXProfileSDK_debug_DEPDIR = $(dir $(@))/$(*F)
$(PhysXProfileSDK_debug_cpp_o): $(PhysXProfileSDK_debug_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling debug $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXProfileSDK_debug_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))))
	@cp $(PhysXProfileSDK_debug_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  rm -f $(PhysXProfileSDK_debug_DEPDIR).d

$(PhysXProfileSDK_debug_c_o): $(PhysXProfileSDK_debug_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling debug $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXProfileSDK_debug_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))))
	@cp $(PhysXProfileSDK_debug_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_debug_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  rm -f $(PhysXProfileSDK_debug_DEPDIR).d

PhysXProfileSDK_checked_hpaths    := 
PhysXProfileSDK_checked_hpaths    += ./../../../Include/foundation
PhysXProfileSDK_checked_hpaths    += ./../../foundation/include
PhysXProfileSDK_checked_hpaths    += ./../../../Include/physxprofilesdk
PhysXProfileSDK_checked_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXProfileSDK_checked_hpaths    += ./../../../Include
PhysXProfileSDK_checked_lpaths    := 
PhysXProfileSDK_checked_defines   := $(PhysXProfileSDK_custom_defines)
PhysXProfileSDK_checked_defines   += PX_PHYSX_STATIC_LIB
PhysXProfileSDK_checked_defines   += NDEBUG
PhysXProfileSDK_checked_defines   += PX_CHECKED
PhysXProfileSDK_checked_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXProfileSDK_checked_libraries := 
PhysXProfileSDK_checked_common_cflags	:= $(PhysXProfileSDK_custom_cflags)
PhysXProfileSDK_checked_common_cflags    += -MMD
PhysXProfileSDK_checked_common_cflags    += $(addprefix -D, $(PhysXProfileSDK_checked_defines))
PhysXProfileSDK_checked_common_cflags    += $(addprefix -I, $(PhysXProfileSDK_checked_hpaths))
PhysXProfileSDK_checked_common_cflags  += -m32
PhysXProfileSDK_checked_cflags	:= $(PhysXProfileSDK_checked_common_cflags)
PhysXProfileSDK_checked_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_checked_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_checked_cflags  += -Wno-long-long
PhysXProfileSDK_checked_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_checked_cflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_checked_cflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
PhysXProfileSDK_checked_cppflags	:= $(PhysXProfileSDK_checked_common_cflags)
PhysXProfileSDK_checked_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_checked_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_checked_cppflags  += -Wno-long-long
PhysXProfileSDK_checked_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_checked_cppflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_checked_cppflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
PhysXProfileSDK_checked_lflags    := $(PhysXProfileSDK_custom_lflags)
PhysXProfileSDK_checked_lflags    += $(addprefix -L, $(PhysXProfileSDK_checked_lpaths))
PhysXProfileSDK_checked_lflags    += -Wl,--start-group $(addprefix -l, $(PhysXProfileSDK_checked_libraries)) -Wl,--end-group
PhysXProfileSDK_checked_lflags  += -lrt
PhysXProfileSDK_checked_lflags  += -m32
PhysXProfileSDK_checked_objsdir  = $(OBJS_DIR)/PhysXProfileSDK_checked
PhysXProfileSDK_checked_cpp_o    = $(addprefix $(PhysXProfileSDK_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_checked_c_o      = $(addprefix $(PhysXProfileSDK_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_checked_obj      = $(PhysXProfileSDK_checked_cpp_o) $(PhysXProfileSDK_checked_c_o)
PhysXProfileSDK_checked_bin      := ./../../../Lib/linux32/libPhysXProfileSDKCHECKED.a

clean_PhysXProfileSDK_checked: 
	@$(ECHO) clean PhysXProfileSDK checked
	@$(RMDIR) $(PhysXProfileSDK_checked_objsdir)
	@$(RMDIR) $(PhysXProfileSDK_checked_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXProfileSDK/checked

build_PhysXProfileSDK_checked: postbuild_PhysXProfileSDK_checked
postbuild_PhysXProfileSDK_checked: mainbuild_PhysXProfileSDK_checked
mainbuild_PhysXProfileSDK_checked: prebuild_PhysXProfileSDK_checked $(PhysXProfileSDK_checked_bin)
prebuild_PhysXProfileSDK_checked:

$(PhysXProfileSDK_checked_bin): $(PhysXProfileSDK_checked_obj) 
	@mkdir -p `dirname ./../../../Lib/linux32/libPhysXProfileSDKCHECKED.a`
	@$(AR) rcs $(PhysXProfileSDK_checked_bin) $(PhysXProfileSDK_checked_obj)
	@$(ECHO) building $@ complete!

PhysXProfileSDK_checked_DEPDIR = $(dir $(@))/$(*F)
$(PhysXProfileSDK_checked_cpp_o): $(PhysXProfileSDK_checked_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling checked $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXProfileSDK_checked_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))))
	@cp $(PhysXProfileSDK_checked_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  rm -f $(PhysXProfileSDK_checked_DEPDIR).d

$(PhysXProfileSDK_checked_c_o): $(PhysXProfileSDK_checked_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling checked $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXProfileSDK_checked_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))))
	@cp $(PhysXProfileSDK_checked_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_checked_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  rm -f $(PhysXProfileSDK_checked_DEPDIR).d

PhysXProfileSDK_profile_hpaths    := 
PhysXProfileSDK_profile_hpaths    += ./../../../Include/foundation
PhysXProfileSDK_profile_hpaths    += ./../../foundation/include
PhysXProfileSDK_profile_hpaths    += ./../../../Include/physxprofilesdk
PhysXProfileSDK_profile_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXProfileSDK_profile_hpaths    += ./../../../Include
PhysXProfileSDK_profile_lpaths    := 
PhysXProfileSDK_profile_defines   := $(PhysXProfileSDK_custom_defines)
PhysXProfileSDK_profile_defines   += PX_PHYSX_STATIC_LIB
PhysXProfileSDK_profile_defines   += NDEBUG
PhysXProfileSDK_profile_defines   += PX_PROFILE
PhysXProfileSDK_profile_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXProfileSDK_profile_libraries := 
PhysXProfileSDK_profile_common_cflags	:= $(PhysXProfileSDK_custom_cflags)
PhysXProfileSDK_profile_common_cflags    += -MMD
PhysXProfileSDK_profile_common_cflags    += $(addprefix -D, $(PhysXProfileSDK_profile_defines))
PhysXProfileSDK_profile_common_cflags    += $(addprefix -I, $(PhysXProfileSDK_profile_hpaths))
PhysXProfileSDK_profile_common_cflags  += -m32
PhysXProfileSDK_profile_cflags	:= $(PhysXProfileSDK_profile_common_cflags)
PhysXProfileSDK_profile_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_profile_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_profile_cflags  += -Wno-long-long
PhysXProfileSDK_profile_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_profile_cflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_profile_cflags  += -O3 -fno-strict-aliasing
PhysXProfileSDK_profile_cppflags	:= $(PhysXProfileSDK_profile_common_cflags)
PhysXProfileSDK_profile_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_profile_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_profile_cppflags  += -Wno-long-long
PhysXProfileSDK_profile_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_profile_cppflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_profile_cppflags  += -O3 -fno-strict-aliasing
PhysXProfileSDK_profile_lflags    := $(PhysXProfileSDK_custom_lflags)
PhysXProfileSDK_profile_lflags    += $(addprefix -L, $(PhysXProfileSDK_profile_lpaths))
PhysXProfileSDK_profile_lflags    += -Wl,--start-group $(addprefix -l, $(PhysXProfileSDK_profile_libraries)) -Wl,--end-group
PhysXProfileSDK_profile_lflags  += -lrt
PhysXProfileSDK_profile_lflags  += -m32
PhysXProfileSDK_profile_objsdir  = $(OBJS_DIR)/PhysXProfileSDK_profile
PhysXProfileSDK_profile_cpp_o    = $(addprefix $(PhysXProfileSDK_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_profile_c_o      = $(addprefix $(PhysXProfileSDK_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_profile_obj      = $(PhysXProfileSDK_profile_cpp_o) $(PhysXProfileSDK_profile_c_o)
PhysXProfileSDK_profile_bin      := ./../../../Lib/linux32/libPhysXProfileSDKPROFILE.a

clean_PhysXProfileSDK_profile: 
	@$(ECHO) clean PhysXProfileSDK profile
	@$(RMDIR) $(PhysXProfileSDK_profile_objsdir)
	@$(RMDIR) $(PhysXProfileSDK_profile_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXProfileSDK/profile

build_PhysXProfileSDK_profile: postbuild_PhysXProfileSDK_profile
postbuild_PhysXProfileSDK_profile: mainbuild_PhysXProfileSDK_profile
mainbuild_PhysXProfileSDK_profile: prebuild_PhysXProfileSDK_profile $(PhysXProfileSDK_profile_bin)
prebuild_PhysXProfileSDK_profile:

$(PhysXProfileSDK_profile_bin): $(PhysXProfileSDK_profile_obj) 
	@mkdir -p `dirname ./../../../Lib/linux32/libPhysXProfileSDKPROFILE.a`
	@$(AR) rcs $(PhysXProfileSDK_profile_bin) $(PhysXProfileSDK_profile_obj)
	@$(ECHO) building $@ complete!

PhysXProfileSDK_profile_DEPDIR = $(dir $(@))/$(*F)
$(PhysXProfileSDK_profile_cpp_o): $(PhysXProfileSDK_profile_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling profile $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXProfileSDK_profile_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))))
	@cp $(PhysXProfileSDK_profile_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  rm -f $(PhysXProfileSDK_profile_DEPDIR).d

$(PhysXProfileSDK_profile_c_o): $(PhysXProfileSDK_profile_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling profile $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXProfileSDK_profile_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))))
	@cp $(PhysXProfileSDK_profile_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_profile_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  rm -f $(PhysXProfileSDK_profile_DEPDIR).d

PhysXProfileSDK_release_hpaths    := 
PhysXProfileSDK_release_hpaths    += ./../../../Include/foundation
PhysXProfileSDK_release_hpaths    += ./../../foundation/include
PhysXProfileSDK_release_hpaths    += ./../../../Include/physxprofilesdk
PhysXProfileSDK_release_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXProfileSDK_release_hpaths    += ./../../../Include
PhysXProfileSDK_release_lpaths    := 
PhysXProfileSDK_release_defines   := $(PhysXProfileSDK_custom_defines)
PhysXProfileSDK_release_defines   += PX_PHYSX_STATIC_LIB
PhysXProfileSDK_release_defines   += NDEBUG
PhysXProfileSDK_release_libraries := 
PhysXProfileSDK_release_common_cflags	:= $(PhysXProfileSDK_custom_cflags)
PhysXProfileSDK_release_common_cflags    += -MMD
PhysXProfileSDK_release_common_cflags    += $(addprefix -D, $(PhysXProfileSDK_release_defines))
PhysXProfileSDK_release_common_cflags    += $(addprefix -I, $(PhysXProfileSDK_release_hpaths))
PhysXProfileSDK_release_common_cflags  += -m32
PhysXProfileSDK_release_cflags	:= $(PhysXProfileSDK_release_common_cflags)
PhysXProfileSDK_release_cflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_release_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_release_cflags  += -Wno-long-long
PhysXProfileSDK_release_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_release_cflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_release_cflags  += -O3 -fno-strict-aliasing
PhysXProfileSDK_release_cppflags	:= $(PhysXProfileSDK_release_common_cflags)
PhysXProfileSDK_release_cppflags  += -Werror -m32 -fPIC -msse2 -mfpmath=sse -malign-double -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden
PhysXProfileSDK_release_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -fdiagnostics-show-option
PhysXProfileSDK_release_cppflags  += -Wno-long-long
PhysXProfileSDK_release_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof -Wno-uninitialized -Wno-attributes -Wno-unused-local-typedefs
PhysXProfileSDK_release_cppflags  += -Wno-unused-parameter -Wno-missing-field-initializers -Wno-ignored-qualifiers
PhysXProfileSDK_release_cppflags  += -O3 -fno-strict-aliasing
PhysXProfileSDK_release_lflags    := $(PhysXProfileSDK_custom_lflags)
PhysXProfileSDK_release_lflags    += $(addprefix -L, $(PhysXProfileSDK_release_lpaths))
PhysXProfileSDK_release_lflags    += -Wl,--start-group $(addprefix -l, $(PhysXProfileSDK_release_libraries)) -Wl,--end-group
PhysXProfileSDK_release_lflags  += -lrt
PhysXProfileSDK_release_lflags  += -m32
PhysXProfileSDK_release_objsdir  = $(OBJS_DIR)/PhysXProfileSDK_release
PhysXProfileSDK_release_cpp_o    = $(addprefix $(PhysXProfileSDK_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXProfileSDK_cppfiles)))))
PhysXProfileSDK_release_c_o      = $(addprefix $(PhysXProfileSDK_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXProfileSDK_cfiles)))))
PhysXProfileSDK_release_obj      = $(PhysXProfileSDK_release_cpp_o) $(PhysXProfileSDK_release_c_o)
PhysXProfileSDK_release_bin      := ./../../../Lib/linux32/libPhysXProfileSDK.a

clean_PhysXProfileSDK_release: 
	@$(ECHO) clean PhysXProfileSDK release
	@$(RMDIR) $(PhysXProfileSDK_release_objsdir)
	@$(RMDIR) $(PhysXProfileSDK_release_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXProfileSDK/release

build_PhysXProfileSDK_release: postbuild_PhysXProfileSDK_release
postbuild_PhysXProfileSDK_release: mainbuild_PhysXProfileSDK_release
mainbuild_PhysXProfileSDK_release: prebuild_PhysXProfileSDK_release $(PhysXProfileSDK_release_bin)
prebuild_PhysXProfileSDK_release:

$(PhysXProfileSDK_release_bin): $(PhysXProfileSDK_release_obj) 
	@mkdir -p `dirname ./../../../Lib/linux32/libPhysXProfileSDK.a`
	@$(AR) rcs $(PhysXProfileSDK_release_bin) $(PhysXProfileSDK_release_obj)
	@$(ECHO) building $@ complete!

PhysXProfileSDK_release_DEPDIR = $(dir $(@))/$(*F)
$(PhysXProfileSDK_release_cpp_o): $(PhysXProfileSDK_release_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling release $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXProfileSDK_release_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))))
	@cp $(PhysXProfileSDK_release_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cppfiles))))).P; \
	  rm -f $(PhysXProfileSDK_release_DEPDIR).d

$(PhysXProfileSDK_release_c_o): $(PhysXProfileSDK_release_objsdir)/%.o:
	@$(ECHO) PhysXProfileSDK: compiling release $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXProfileSDK_release_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))))
	@cp $(PhysXProfileSDK_release_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXProfileSDK_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXProfileSDK/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXProfileSDK_release_objsdir),, $@))), $(PhysXProfileSDK_cfiles))))).P; \
	  rm -f $(PhysXProfileSDK_release_DEPDIR).d

clean_PhysXProfileSDK:  clean_PhysXProfileSDK_debug clean_PhysXProfileSDK_checked clean_PhysXProfileSDK_profile clean_PhysXProfileSDK_release
	@$(RMDIR) $(DEPSDIR)/PhysXProfileSDK
