################################################################################
#
# Common build script
#
################################################################################

.SUFFIXES : .cu .cu_dbg_o .c_dbg_o .cpp_dbg_o .cu_rel_o .c_rel_o .cpp_rel_o .cubin

CUDA_INSTALL_PATH ?= /usr/local/cuda
CUDA_SDK_PATH ?= /usr/local/cuda/

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= 
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/bin
BINDIR     ?= $(ROOTBINDIR)/linux
ROOTOBJDIR ?= $(ROOTDIR)/obj
ROOTSODIR  ?= $(ROOTDIR)/lib
SODIR      ?= $(ROOTSODIR)/linux

CUDALIBSUFFIX := lib64
LIBDIR := $(CUDA_SDK_PATH)/lib64
COMMONDIR := /usr/local/cuda/samples/common
ACMLDIR ?= /opt/acml5.3.1/gfortran64/

GCCDIR ?= /usr/bin
# Compilers
#GCCDIR ?= /projects/grail/library/uns/bin/
NVCC       := nvcc 
CXX        := $(GCCDIR)/g++
CC         := $(GCCDIR)/gcc
LINK       := $(GCCDIR)/g++

# Includes
INCLUDES  += -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc -I$(ROOTDIR)/include -I$(ACMLDIR)/include -I../include -I/usr/include/atlas/

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m64

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
OPENGLLIB := -lGL -lGLU -lGLEW
GLEWLIBPATH := /usr/local/cuda/samples/common/lib/linux/x86_64/

UNSLIB := /usr/lib/
LAPACKLIB := /usr/lib/

# Libs
LIB       := -L/usr/lib/ -L./ -L../ -L/usr/lib/atlas/ -L$(CUDA_INSTALL_PATH)/$(CUDALIBSUFFIX) -L$(ACMLDIR)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -L$(GLEWLIBPATH) -L$(UNSLIB) -L$(LAPACKLIB) -lacml -lcuda -lcudart -lcublas -lblas -llapack -lgfortran -L../stencilMatrixMultiply/lib/linux/release ${OPENGLLIB} ${LIB}

# Warning flags
CXXWARN_FLAGS := \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
        -Wno-unused-result \
        -Wno-sign-compare \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
        -Wno-sign-compare \
	-Wmain \

# Compiler-specific flags
NVCCFLAGS := -Xptxas -v -maxrregcount 64
CXXFLAGS  := $(CXXWARN_FLAGS)
CFLAGS    := $(CWARN_FLAGS)

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3 
	BINSUBDIR   := release
	LIBSUFFIX   :=
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

ifeq ($(shared),1)
	NVCCFLAGS  += -Xcompiler -fPIC
	CFLAGS += -fPIC
	CXXFLAGS += -fPIC
	LINK += -shared
endif

# append optional arch/SM version flags (such as -arch sm_20)
SMVERSIONFLAGS ?= -arch sm_35
NVCCFLAGS += $(SMVERSIONFLAGS)

# architecture flag for cubin build
CUBIN_ARCH_FLAG := -m64

# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
OPENGLLIB := -lGL -lGLU -lGLEW
GLEWLIBPATH := /usr/local/cuda/samples/common/lib/linux/x86_64/
UNSLIB := /usr/lib/
LAPACKLIB := /usr/lib/

ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl$(LIBSUFFIX)
endif

# Libs
LIB       := -L/usr/lib64/ -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -L$(GLEWLIBPATH) -L$(UNSLIB) -L$(LAPACKLIB) -lcuda -lcudart ${OPENGLLIB} $(PARAMGLLIB) ${LIB}


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
	LINKLINE  = ar qv $(TARGET) $(OBJS) 
else
	LIB += $(LIBSUFFIX)
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS		+= -D__DEVICE_EMULATION__
		CFLAGS			+= -D__DEVICE_EMULATION__
	endif
	ifeq ($(shared),1)
		TARGETDIR := $(SODIR)/$(BINSUBDIR)
		TARGET := $(TARGETDIR)/lib$(EXECUTABLE).so
  else
		TARGETDIR := $(BINDIR)/$(BINSUBDIR)
		TARGET    := $(TARGETDIR)/$(EXECUTABLE)
	endif
	ifndef NOLINK
		LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LINKOBJS) $(LIB)
	else
		LINKLINE  =
	endif
endif

VERBOSE := @

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif

ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

###########################################
# Check for atomics support
############################################
ifeq ($(findstring sm_10, $(SMVERSIONFLAGS)),sm_10)
	NVCCFLAGS += -D __NO_ATOMIC 
endif
ifeq ($(findstring sm_11, $(SMVERSIONFLAGS)),sm_11)
	NVCCFLAGS += -D __NO_ATOMIC 
endif



# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS)

# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

#ifeq ($(nvcc_warn_verbose),1)
#	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
#	NVCCFLAGS += --compiler-options -fno-strict-aliasing
#endif

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)

LINKOBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp_o,$(notdir $(LINKCCFILES)))
LINKOBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(LINKCFILES)))
LINKOBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(LINKCUFILES)))

OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp_o,$(notdir $(CCFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c_o,$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu_o,$(notdir $(CUFILES)))

################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(SRCDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c_o : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp_o : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu_o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $@ -c $< $(NVCCFLAGS)

$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) -o $@ -cubin $< $(NVCCFLAGS)

$(TARGET): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)

cubindirectory:
	@mkdir -p $(CUBINDIR)

makedirectories:
	@mkdir -p $(LIBDIR)
	@mkdir -p $(OBJDIR)
	@mkdir -p $(TARGETDIR)


tidy :
	@find | egrep "#" | xargs rm -f
	@find | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)

clobber : clean
	rm -rf $(ROOTOBJDIR)
