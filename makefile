CC=g++
CXXFLAGS = -std=c++11 -g -mfma -fopenmp -I /home/bbensaid/Documents/libC/eigen-3.4.0 -I /home/bbensaid/Documents/libC/shaman/PREFIX/include -I /home/bbensaid/Documents/libC/EigenRand-0.4.0alpha -D SHAMAN_UNSTABLE_BRANCH
LDFLAGS = -lshaman
LDLIBS = -L /home/bbensaid/Documents/libC/shaman/PREFIX/lib
EXEC = training
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)
	 
main.o: init.h data.h training.h test.h tirage.h Straining.h Stest.h
utilities.o: eigenExtension.h
init.o: eigenExtension.h
propagation.o: activations.h perte.h eigenExtension.h
classic.o: propagation.h eigenExtension.h utilities.h
Sclassic.o: propagation.h eigenExtension.h 
Sperso.o: propagation.h eigenExtension.h Sclassic.h
LMs.o: propagation.h scaling.h utilities.h eigenExtension.h
perso.o: propagation.h eigenExtension.h classic.h
essai.o: propagation.h eigenExtension.h classic.h utilities.h
training.o: classic.h LMs.h perso.h
Straining.o: Sclassic.h Sperso.h
test.o: init.h training.h utilities.h
Stest.o: init.h Straining.h utilities.h
Stirage.o: init.h propagation.h Straining.h eigenExtension.h
 
%.o: %.c
	$(CC) -o $@ -c $< $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)

clean:
	rm -f *.o core

mrproper: clean
	rm -f $(EXEC)
