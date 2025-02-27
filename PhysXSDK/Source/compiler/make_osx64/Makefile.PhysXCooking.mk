# Makefile generated by XPJ for osx64
-include Makefile.custom
ProjectName = PhysXCooking
PhysXCooking_custom_cflags := -isysroot $(APPLE_OSX_SDK_CURRENT_VERSION)
PhysXCooking_custom_lflags := -isysroot $(APPLE_OSX_SDK_CURRENT_VERSION)
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/Adjacencies.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/Cooking.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/CookingUtils.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/EdgeList.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/InternalTriangleMeshBuilder.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/MeshCleaner.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/mesh/HeightFieldCooking.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/mesh/RTreeCooking.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/mesh/TriangleMeshBuilder.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/convex/BigConvexDataBuilder.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/convex/ConvexHullBuilder.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/convex/ConvexMeshBuilder.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/convex/VolumeIntegration.cpp
PhysXCooking_cppfiles   += ./../../PhysXCooking/src/convex/hulllib.cpp

PhysXCooking_cpp_release_dep    = $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXCooking_cppfiles)))))
PhysXCooking_c_release_dep      = $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXCooking_cfiles)))))
PhysXCooking_release_dep      = $(PhysXCooking_cpp_release_dep) $(PhysXCooking_c_release_dep)
-include $(PhysXCooking_release_dep)
PhysXCooking_cpp_debug_dep    = $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXCooking_cppfiles)))))
PhysXCooking_c_debug_dep      = $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXCooking_cfiles)))))
PhysXCooking_debug_dep      = $(PhysXCooking_cpp_debug_dep) $(PhysXCooking_c_debug_dep)
-include $(PhysXCooking_debug_dep)
PhysXCooking_cpp_checked_dep    = $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXCooking_cppfiles)))))
PhysXCooking_c_checked_dep      = $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXCooking_cfiles)))))
PhysXCooking_checked_dep      = $(PhysXCooking_cpp_checked_dep) $(PhysXCooking_c_checked_dep)
-include $(PhysXCooking_checked_dep)
PhysXCooking_cpp_profile_dep    = $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.P, $(PhysXCooking_cppfiles)))))
PhysXCooking_c_profile_dep      = $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.P, $(PhysXCooking_cfiles)))))
PhysXCooking_profile_dep      = $(PhysXCooking_cpp_profile_dep) $(PhysXCooking_c_profile_dep)
-include $(PhysXCooking_profile_dep)
PhysXCooking_release_hpaths    := 
PhysXCooking_release_hpaths    += ./../../../Include/foundation
PhysXCooking_release_hpaths    += ./../../foundation/include
PhysXCooking_release_hpaths    += ./../../../Include/physxprofilesdk
PhysXCooking_release_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXCooking_release_hpaths    += ./../../../Include/common
PhysXCooking_release_hpaths    += ./../../../Include/geometry
PhysXCooking_release_hpaths    += ./../../../Include/cloth
PhysXCooking_release_hpaths    += ./../../../Include/cooking
PhysXCooking_release_hpaths    += ./../../../Include
PhysXCooking_release_hpaths    += ./../../Common/src
PhysXCooking_release_hpaths    += ./../../GeomUtils/headers
PhysXCooking_release_hpaths    += ./../../GeomUtils/src
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/contact
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/common
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/convex
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/distance
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/sweep
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/gjk
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/intersection
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/mesh
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/hf
PhysXCooking_release_hpaths    += ./../../GeomUtils/src/pcm
PhysXCooking_release_hpaths    += ./../../PhysXCooking/src
PhysXCooking_release_hpaths    += ./../../PhysXCooking/src/mesh
PhysXCooking_release_hpaths    += ./../../PhysXCooking/src/convex
PhysXCooking_release_hpaths    += ./../../PhysXExtensions/src
PhysXCooking_release_lpaths    := 
PhysXCooking_release_lpaths    += ./../../../Lib/osx64
PhysXCooking_release_defines   := $(PhysXCooking_custom_defines)
PhysXCooking_release_defines   += PX_PHYSX_STATIC_LIB
PhysXCooking_release_defines   += PX_COOKING
PhysXCooking_release_defines   += NDEBUG
PhysXCooking_release_libraries := 
PhysXCooking_release_libraries += PhysX3Common
PhysXCooking_release_libraries += PhysX3Extensions
PhysXCooking_release_common_cflags	:= $(PhysXCooking_custom_cflags)
PhysXCooking_release_common_cflags    += -MMD
PhysXCooking_release_common_cflags    += $(addprefix -D, $(PhysXCooking_release_defines))
PhysXCooking_release_common_cflags    += $(addprefix -I, $(PhysXCooking_release_hpaths))
PhysXCooking_release_cflags	:= $(PhysXCooking_release_common_cflags)
PhysXCooking_release_cflags  += -arch x86_64
PhysXCooking_release_cflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_release_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_release_cflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_release_cflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_release_cflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_release_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_release_cflags  += -Wno-unused-parameter
PhysXCooking_release_cflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_release_cflags  += -O3 -fno-strict-aliasing
PhysXCooking_release_cppflags	:= $(PhysXCooking_release_common_cflags)
PhysXCooking_release_cppflags  += -arch x86_64
PhysXCooking_release_cppflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_release_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_release_cppflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_release_cppflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_release_cppflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_release_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_release_cppflags  += -Wno-unused-parameter
PhysXCooking_release_cppflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_release_cppflags  += -O3 -fno-strict-aliasing
PhysXCooking_release_lflags    := $(PhysXCooking_custom_lflags)
PhysXCooking_release_lflags    += $(addprefix -L, $(PhysXCooking_release_lpaths))
PhysXCooking_release_lflags  += $(addprefix -l, $(PhysXCooking_release_libraries))
PhysXCooking_release_lflags  += -arch x86_64
PhysXCooking_release_objsdir  = $(OBJS_DIR)/PhysXCooking_release
PhysXCooking_release_cpp_o    = $(addprefix $(PhysXCooking_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXCooking_cppfiles)))))
PhysXCooking_release_c_o      = $(addprefix $(PhysXCooking_release_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXCooking_cfiles)))))
PhysXCooking_release_obj      = $(PhysXCooking_release_cpp_o) $(PhysXCooking_release_c_o)
PhysXCooking_release_bin      := ./../../../Lib/osx64/libPhysX3Cooking.a

clean_PhysXCooking_release: 
	@$(ECHO) clean PhysXCooking release
	@$(RMDIR) $(PhysXCooking_release_objsdir)
	@$(RMDIR) $(PhysXCooking_release_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXCooking/release

build_PhysXCooking_release: postbuild_PhysXCooking_release
postbuild_PhysXCooking_release: mainbuild_PhysXCooking_release
mainbuild_PhysXCooking_release: prebuild_PhysXCooking_release $(PhysXCooking_release_bin)
prebuild_PhysXCooking_release:

$(PhysXCooking_release_bin): $(PhysXCooking_release_obj) build_PhysXCommon_release build_PhysXExtensions_release 
	@mkdir -p `dirname ./../../../Lib/osx64/libPhysX3Cooking.a`
	@$(AR) rcs $(PhysXCooking_release_bin) $(PhysXCooking_release_obj)
	@$(ECHO) building $@ complete!

PhysXCooking_release_DEPDIR = $(dir $(@))/$(*F)
$(PhysXCooking_release_cpp_o): $(PhysXCooking_release_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling release $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXCooking_release_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cppfiles))))))
	@cp $(PhysXCooking_release_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  rm -f $(PhysXCooking_release_DEPDIR).d

$(PhysXCooking_release_c_o): $(PhysXCooking_release_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling release $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXCooking_release_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cfiles))))))
	@cp $(PhysXCooking_release_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_release_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/release/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_release_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  rm -f $(PhysXCooking_release_DEPDIR).d

PhysXCooking_debug_hpaths    := 
PhysXCooking_debug_hpaths    += ./../../../Include/foundation
PhysXCooking_debug_hpaths    += ./../../foundation/include
PhysXCooking_debug_hpaths    += ./../../../Include/physxprofilesdk
PhysXCooking_debug_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXCooking_debug_hpaths    += ./../../../Include/common
PhysXCooking_debug_hpaths    += ./../../../Include/geometry
PhysXCooking_debug_hpaths    += ./../../../Include/cloth
PhysXCooking_debug_hpaths    += ./../../../Include/cooking
PhysXCooking_debug_hpaths    += ./../../../Include
PhysXCooking_debug_hpaths    += ./../../Common/src
PhysXCooking_debug_hpaths    += ./../../GeomUtils/headers
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/contact
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/common
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/convex
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/distance
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/sweep
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/gjk
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/intersection
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/mesh
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/hf
PhysXCooking_debug_hpaths    += ./../../GeomUtils/src/pcm
PhysXCooking_debug_hpaths    += ./../../PhysXCooking/src
PhysXCooking_debug_hpaths    += ./../../PhysXCooking/src/mesh
PhysXCooking_debug_hpaths    += ./../../PhysXCooking/src/convex
PhysXCooking_debug_hpaths    += ./../../PhysXExtensions/src
PhysXCooking_debug_lpaths    := 
PhysXCooking_debug_lpaths    += ./../../../Lib/osx64
PhysXCooking_debug_defines   := $(PhysXCooking_custom_defines)
PhysXCooking_debug_defines   += PX_PHYSX_STATIC_LIB
PhysXCooking_debug_defines   += PX_COOKING
PhysXCooking_debug_defines   += _DEBUG
PhysXCooking_debug_defines   += PX_DEBUG
PhysXCooking_debug_defines   += PX_CHECKED
PhysXCooking_debug_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXCooking_debug_libraries := 
PhysXCooking_debug_libraries += PhysX3CommonDEBUG
PhysXCooking_debug_libraries += PhysX3ExtensionsDEBUG
PhysXCooking_debug_common_cflags	:= $(PhysXCooking_custom_cflags)
PhysXCooking_debug_common_cflags    += -MMD
PhysXCooking_debug_common_cflags    += $(addprefix -D, $(PhysXCooking_debug_defines))
PhysXCooking_debug_common_cflags    += $(addprefix -I, $(PhysXCooking_debug_hpaths))
PhysXCooking_debug_cflags	:= $(PhysXCooking_debug_common_cflags)
PhysXCooking_debug_cflags  += -arch x86_64
PhysXCooking_debug_cflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_debug_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_debug_cflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_debug_cflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_debug_cflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_debug_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_debug_cflags  += -Wno-unused-parameter
PhysXCooking_debug_cflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_debug_cflags  += -g3 -gdwarf-2 -O0
PhysXCooking_debug_cppflags	:= $(PhysXCooking_debug_common_cflags)
PhysXCooking_debug_cppflags  += -arch x86_64
PhysXCooking_debug_cppflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_debug_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_debug_cppflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_debug_cppflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_debug_cppflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_debug_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_debug_cppflags  += -Wno-unused-parameter
PhysXCooking_debug_cppflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_debug_cppflags  += -g3 -gdwarf-2 -O0
PhysXCooking_debug_lflags    := $(PhysXCooking_custom_lflags)
PhysXCooking_debug_lflags    += $(addprefix -L, $(PhysXCooking_debug_lpaths))
PhysXCooking_debug_lflags  += $(addprefix -l, $(PhysXCooking_debug_libraries))
PhysXCooking_debug_lflags  += -arch x86_64
PhysXCooking_debug_objsdir  = $(OBJS_DIR)/PhysXCooking_debug
PhysXCooking_debug_cpp_o    = $(addprefix $(PhysXCooking_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXCooking_cppfiles)))))
PhysXCooking_debug_c_o      = $(addprefix $(PhysXCooking_debug_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXCooking_cfiles)))))
PhysXCooking_debug_obj      = $(PhysXCooking_debug_cpp_o) $(PhysXCooking_debug_c_o)
PhysXCooking_debug_bin      := ./../../../Lib/osx64/libPhysX3CookingDEBUG.a

clean_PhysXCooking_debug: 
	@$(ECHO) clean PhysXCooking debug
	@$(RMDIR) $(PhysXCooking_debug_objsdir)
	@$(RMDIR) $(PhysXCooking_debug_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXCooking/debug

build_PhysXCooking_debug: postbuild_PhysXCooking_debug
postbuild_PhysXCooking_debug: mainbuild_PhysXCooking_debug
mainbuild_PhysXCooking_debug: prebuild_PhysXCooking_debug $(PhysXCooking_debug_bin)
prebuild_PhysXCooking_debug:

$(PhysXCooking_debug_bin): $(PhysXCooking_debug_obj) build_PhysXCommon_debug build_PhysXExtensions_debug 
	@mkdir -p `dirname ./../../../Lib/osx64/libPhysX3CookingDEBUG.a`
	@$(AR) rcs $(PhysXCooking_debug_bin) $(PhysXCooking_debug_obj)
	@$(ECHO) building $@ complete!

PhysXCooking_debug_DEPDIR = $(dir $(@))/$(*F)
$(PhysXCooking_debug_cpp_o): $(PhysXCooking_debug_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling debug $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXCooking_debug_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cppfiles))))))
	@cp $(PhysXCooking_debug_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  rm -f $(PhysXCooking_debug_DEPDIR).d

$(PhysXCooking_debug_c_o): $(PhysXCooking_debug_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling debug $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXCooking_debug_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cfiles))))))
	@cp $(PhysXCooking_debug_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_debug_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/debug/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_debug_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  rm -f $(PhysXCooking_debug_DEPDIR).d

PhysXCooking_checked_hpaths    := 
PhysXCooking_checked_hpaths    += ./../../../Include/foundation
PhysXCooking_checked_hpaths    += ./../../foundation/include
PhysXCooking_checked_hpaths    += ./../../../Include/physxprofilesdk
PhysXCooking_checked_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXCooking_checked_hpaths    += ./../../../Include/common
PhysXCooking_checked_hpaths    += ./../../../Include/geometry
PhysXCooking_checked_hpaths    += ./../../../Include/cloth
PhysXCooking_checked_hpaths    += ./../../../Include/cooking
PhysXCooking_checked_hpaths    += ./../../../Include
PhysXCooking_checked_hpaths    += ./../../Common/src
PhysXCooking_checked_hpaths    += ./../../GeomUtils/headers
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/contact
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/common
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/convex
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/distance
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/sweep
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/gjk
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/intersection
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/mesh
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/hf
PhysXCooking_checked_hpaths    += ./../../GeomUtils/src/pcm
PhysXCooking_checked_hpaths    += ./../../PhysXCooking/src
PhysXCooking_checked_hpaths    += ./../../PhysXCooking/src/mesh
PhysXCooking_checked_hpaths    += ./../../PhysXCooking/src/convex
PhysXCooking_checked_hpaths    += ./../../PhysXExtensions/src
PhysXCooking_checked_lpaths    := 
PhysXCooking_checked_lpaths    += ./../../../Lib/osx64
PhysXCooking_checked_defines   := $(PhysXCooking_custom_defines)
PhysXCooking_checked_defines   += PX_PHYSX_STATIC_LIB
PhysXCooking_checked_defines   += PX_COOKING
PhysXCooking_checked_defines   += NDEBUG
PhysXCooking_checked_defines   += PX_CHECKED
PhysXCooking_checked_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXCooking_checked_libraries := 
PhysXCooking_checked_libraries += PhysX3CommonCHECKED
PhysXCooking_checked_libraries += PhysX3ExtensionsCHECKED
PhysXCooking_checked_common_cflags	:= $(PhysXCooking_custom_cflags)
PhysXCooking_checked_common_cflags    += -MMD
PhysXCooking_checked_common_cflags    += $(addprefix -D, $(PhysXCooking_checked_defines))
PhysXCooking_checked_common_cflags    += $(addprefix -I, $(PhysXCooking_checked_hpaths))
PhysXCooking_checked_cflags	:= $(PhysXCooking_checked_common_cflags)
PhysXCooking_checked_cflags  += -arch x86_64
PhysXCooking_checked_cflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_checked_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_checked_cflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_checked_cflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_checked_cflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_checked_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_checked_cflags  += -Wno-unused-parameter
PhysXCooking_checked_cflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_checked_cflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
PhysXCooking_checked_cppflags	:= $(PhysXCooking_checked_common_cflags)
PhysXCooking_checked_cppflags  += -arch x86_64
PhysXCooking_checked_cppflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_checked_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_checked_cppflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_checked_cppflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_checked_cppflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_checked_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_checked_cppflags  += -Wno-unused-parameter
PhysXCooking_checked_cppflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_checked_cppflags  += -g3 -gdwarf-2 -O3 -fno-strict-aliasing
PhysXCooking_checked_lflags    := $(PhysXCooking_custom_lflags)
PhysXCooking_checked_lflags    += $(addprefix -L, $(PhysXCooking_checked_lpaths))
PhysXCooking_checked_lflags  += $(addprefix -l, $(PhysXCooking_checked_libraries))
PhysXCooking_checked_lflags  += -arch x86_64
PhysXCooking_checked_objsdir  = $(OBJS_DIR)/PhysXCooking_checked
PhysXCooking_checked_cpp_o    = $(addprefix $(PhysXCooking_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXCooking_cppfiles)))))
PhysXCooking_checked_c_o      = $(addprefix $(PhysXCooking_checked_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXCooking_cfiles)))))
PhysXCooking_checked_obj      = $(PhysXCooking_checked_cpp_o) $(PhysXCooking_checked_c_o)
PhysXCooking_checked_bin      := ./../../../Lib/osx64/libPhysX3CookingCHECKED.a

clean_PhysXCooking_checked: 
	@$(ECHO) clean PhysXCooking checked
	@$(RMDIR) $(PhysXCooking_checked_objsdir)
	@$(RMDIR) $(PhysXCooking_checked_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXCooking/checked

build_PhysXCooking_checked: postbuild_PhysXCooking_checked
postbuild_PhysXCooking_checked: mainbuild_PhysXCooking_checked
mainbuild_PhysXCooking_checked: prebuild_PhysXCooking_checked $(PhysXCooking_checked_bin)
prebuild_PhysXCooking_checked:

$(PhysXCooking_checked_bin): $(PhysXCooking_checked_obj) build_PhysXCommon_checked build_PhysXExtensions_checked 
	@mkdir -p `dirname ./../../../Lib/osx64/libPhysX3CookingCHECKED.a`
	@$(AR) rcs $(PhysXCooking_checked_bin) $(PhysXCooking_checked_obj)
	@$(ECHO) building $@ complete!

PhysXCooking_checked_DEPDIR = $(dir $(@))/$(*F)
$(PhysXCooking_checked_cpp_o): $(PhysXCooking_checked_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling checked $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXCooking_checked_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cppfiles))))))
	@cp $(PhysXCooking_checked_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  rm -f $(PhysXCooking_checked_DEPDIR).d

$(PhysXCooking_checked_c_o): $(PhysXCooking_checked_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling checked $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXCooking_checked_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cfiles))))))
	@cp $(PhysXCooking_checked_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_checked_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/checked/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_checked_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  rm -f $(PhysXCooking_checked_DEPDIR).d

PhysXCooking_profile_hpaths    := 
PhysXCooking_profile_hpaths    += ./../../../Include/foundation
PhysXCooking_profile_hpaths    += ./../../foundation/include
PhysXCooking_profile_hpaths    += ./../../../Include/physxprofilesdk
PhysXCooking_profile_hpaths    += ./../../../Include/physxvisualdebuggersdk
PhysXCooking_profile_hpaths    += ./../../../Include/common
PhysXCooking_profile_hpaths    += ./../../../Include/geometry
PhysXCooking_profile_hpaths    += ./../../../Include/cloth
PhysXCooking_profile_hpaths    += ./../../../Include/cooking
PhysXCooking_profile_hpaths    += ./../../../Include
PhysXCooking_profile_hpaths    += ./../../Common/src
PhysXCooking_profile_hpaths    += ./../../GeomUtils/headers
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/contact
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/common
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/convex
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/distance
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/sweep
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/gjk
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/intersection
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/mesh
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/hf
PhysXCooking_profile_hpaths    += ./../../GeomUtils/src/pcm
PhysXCooking_profile_hpaths    += ./../../PhysXCooking/src
PhysXCooking_profile_hpaths    += ./../../PhysXCooking/src/mesh
PhysXCooking_profile_hpaths    += ./../../PhysXCooking/src/convex
PhysXCooking_profile_hpaths    += ./../../PhysXExtensions/src
PhysXCooking_profile_lpaths    := 
PhysXCooking_profile_lpaths    += ./../../../Lib/osx64
PhysXCooking_profile_defines   := $(PhysXCooking_custom_defines)
PhysXCooking_profile_defines   += PX_PHYSX_STATIC_LIB
PhysXCooking_profile_defines   += PX_COOKING
PhysXCooking_profile_defines   += NDEBUG
PhysXCooking_profile_defines   += PX_PROFILE
PhysXCooking_profile_defines   += PX_SUPPORT_VISUAL_DEBUGGER
PhysXCooking_profile_libraries := 
PhysXCooking_profile_libraries += PhysX3CommonPROFILE
PhysXCooking_profile_libraries += PhysX3ExtensionsPROFILE
PhysXCooking_profile_common_cflags	:= $(PhysXCooking_custom_cflags)
PhysXCooking_profile_common_cflags    += -MMD
PhysXCooking_profile_common_cflags    += $(addprefix -D, $(PhysXCooking_profile_defines))
PhysXCooking_profile_common_cflags    += $(addprefix -I, $(PhysXCooking_profile_hpaths))
PhysXCooking_profile_cflags	:= $(PhysXCooking_profile_common_cflags)
PhysXCooking_profile_cflags  += -arch x86_64
PhysXCooking_profile_cflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_profile_cflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_profile_cflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_profile_cflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_profile_cflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_profile_cflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_profile_cflags  += -Wno-unused-parameter
PhysXCooking_profile_cflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_profile_cflags  += -O3 -fno-strict-aliasing
PhysXCooking_profile_cppflags	:= $(PhysXCooking_profile_common_cflags)
PhysXCooking_profile_cppflags  += -arch x86_64
PhysXCooking_profile_cppflags  += -pipe -mmacosx-version-min=10.5 -msse2 -ffast-math -fno-exceptions -fno-rtti -fvisibility=hidden -fvisibility-inlines-hidden -Werror
PhysXCooking_profile_cppflags  += -Wall -Wextra -Wstrict-aliasing=2 -Weverything
PhysXCooking_profile_cppflags  += -Wno-pedantic -Wno-unknown-warning-option
PhysXCooking_profile_cppflags  += -Wno-long-long -Wno-newline-eof -Wno-extended-offsetof
PhysXCooking_profile_cppflags  += -Wno-float-equal -Wno-documentation-deprecated-sync -Wno-conversion -Wno-weak-vtables -Wno-unreachable-code -Wno-format-nonliteral -Wno-cast-align -Wno-documentation -Wno-covered-switch-default -Wno-documentation-unknown-command -Wno-padded
PhysXCooking_profile_cppflags  += -Wno-unknown-pragmas -Wno-invalid-offsetof
PhysXCooking_profile_cppflags  += -Wno-unused-parameter
PhysXCooking_profile_cppflags  += -Wno-global-constructors -Wno-exit-time-destructors -Wno-weak-template-vtables -Wno-shift-sign-overflow -Wno-missing-noreturn -Wno-missing-variable-declarations -Wno-switch-enum -Wno-undef -Wno-unused-macros -Wno-c99-extensions -Wno-missing-prototypes -Wno-shadow -Wno-unused-member-function -Wno-used-but-marked-unused -Wno-header-hygiene -Wno-variadic-macros
PhysXCooking_profile_cppflags  += -O3 -fno-strict-aliasing
PhysXCooking_profile_lflags    := $(PhysXCooking_custom_lflags)
PhysXCooking_profile_lflags    += $(addprefix -L, $(PhysXCooking_profile_lpaths))
PhysXCooking_profile_lflags  += $(addprefix -l, $(PhysXCooking_profile_libraries))
PhysXCooking_profile_lflags  += -arch x86_64
PhysXCooking_profile_objsdir  = $(OBJS_DIR)/PhysXCooking_profile
PhysXCooking_profile_cpp_o    = $(addprefix $(PhysXCooking_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.cpp, %.cpp.o, $(PhysXCooking_cppfiles)))))
PhysXCooking_profile_c_o      = $(addprefix $(PhysXCooking_profile_objsdir)/, $(subst ./, , $(subst ../, , $(patsubst %.c, %.c.o, $(PhysXCooking_cfiles)))))
PhysXCooking_profile_obj      = $(PhysXCooking_profile_cpp_o) $(PhysXCooking_profile_c_o)
PhysXCooking_profile_bin      := ./../../../Lib/osx64/libPhysX3CookingPROFILE.a

clean_PhysXCooking_profile: 
	@$(ECHO) clean PhysXCooking profile
	@$(RMDIR) $(PhysXCooking_profile_objsdir)
	@$(RMDIR) $(PhysXCooking_profile_bin)
	@$(RMDIR) $(DEPSDIR)/PhysXCooking/profile

build_PhysXCooking_profile: postbuild_PhysXCooking_profile
postbuild_PhysXCooking_profile: mainbuild_PhysXCooking_profile
mainbuild_PhysXCooking_profile: prebuild_PhysXCooking_profile $(PhysXCooking_profile_bin)
prebuild_PhysXCooking_profile:

$(PhysXCooking_profile_bin): $(PhysXCooking_profile_obj) build_PhysXCommon_profile build_PhysXExtensions_profile 
	@mkdir -p `dirname ./../../../Lib/osx64/libPhysX3CookingPROFILE.a`
	@$(AR) rcs $(PhysXCooking_profile_bin) $(PhysXCooking_profile_obj)
	@$(ECHO) building $@ complete!

PhysXCooking_profile_DEPDIR = $(dir $(@))/$(*F)
$(PhysXCooking_profile_cpp_o): $(PhysXCooking_profile_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling profile $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cppfiles))...
	@mkdir -p $(dir $(@))
	@$(CXX) $(PhysXCooking_profile_cppflags) -c $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cppfiles)) -o $@
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cppfiles))))))
	@cp $(PhysXCooking_profile_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .cpp.o,.cpp, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cppfiles))))).P; \
	  rm -f $(PhysXCooking_profile_DEPDIR).d

$(PhysXCooking_profile_c_o): $(PhysXCooking_profile_objsdir)/%.o:
	@$(ECHO) PhysXCooking: compiling profile $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cfiles))...
	@mkdir -p $(dir $(@))
	@$(CC) $(PhysXCooking_profile_cflags) -c $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cfiles)) -o $@ 
	@mkdir -p $(dir $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cfiles))))))
	@cp $(PhysXCooking_profile_DEPDIR).d $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
		-e '/^$$/ d' -e 's/$$/ :/' < $(PhysXCooking_profile_DEPDIR).d >> $(addprefix $(DEPSDIR)/PhysXCooking/profile/, $(subst ./, , $(subst ../, , $(filter %$(strip $(subst .c.o,.c, $(subst $(PhysXCooking_profile_objsdir),, $@))), $(PhysXCooking_cfiles))))).P; \
	  rm -f $(PhysXCooking_profile_DEPDIR).d

clean_PhysXCooking:  clean_PhysXCooking_release clean_PhysXCooking_debug clean_PhysXCooking_checked clean_PhysXCooking_profile
	@$(RMDIR) $(DEPSDIR)/PhysXCooking
