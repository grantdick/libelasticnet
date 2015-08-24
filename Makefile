AR:=ar
CC:=gcc
SRCDIR:=src
DEMDIR:=demo
OBJDIR:=build
INCDIR:=include
BINDIR:=dist
LIBDIR:=$(BINDIR)

INCS:=$(wildcard $(SRCDIR)/*.h)
OBJS:=$(subst $(SRCDIR)/,$(OBJDIR)/,$(patsubst %.c,%.o,$(wildcard $(SRCDIR)/*.c)))
DEMO_OBJS=$(OBJDIR)/mt19937ar.o $(OBJDIR)/readline.o

CFLAGS:=-std=gnu99 -Wall -pedantic -march=native -O3 -g
IFLAGS:=-I$(INCDIR)
LFLAGS:=-L$(LIBDIR) -lelasticnet -lm

INC:=$(SRCDIR)/elasticnet.h

LIB:=$(LIBDIR)/libelasticnet.a
BIN:=$(BINDIR)/cvelnet

all: $(LIB) $(BIN)

lib: $(LIB)

$(LIBDIR)/libelasticnet.a: $(OBJS) $(INCS)
	@echo creating library $@ from $^
	@mkdir -p $(BINDIR)
	@$(AR) -r $@ $(OBJS)
	@echo copying headers to $(INCDIR)
	@mkdir -p $(INCDIR)
	@cp $(INC) $(INCDIR)

$(BINDIR)/cvelnet: $(DEMO_OBJS) $(OBJDIR)/data.o $(OBJDIR)/main.o $(LIB)
	@echo linking $@ from $^
	@$(CC) $(CFLAGS) $^ -o $@ $(LFLAGS)

$(OBJDIR)/%.o : $(SRCDIR)/%.c $(INCS)
	@echo compiling $< into $@
	@mkdir -p $(OBJDIR)
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

$(OBJDIR)/%.o : $(DEMDIR)/%.c $(wildcard $(DEMDIR)/*.h) $(INC)
	@echo compiling $< into $@
	@$(CC) $(CFLAGS) $(IFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJDIR)

nuke: clean
	@rm -rf $(INCDIR) $(BINDIR) $(LIBDIR)

strip: all
	@echo running strip on $(BIN)
	@strip $(BIN)
