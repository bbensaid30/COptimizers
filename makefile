CC=g++
CXXFLAGS = -std=c++11 -g -mfma -fopenmp -I /home/bbensaid/Documents/libC/eigen-3.4.0 -I /home/bbensaid/Documents/libC/EigenRand-0.4.0alpha
LDFLAGS = 
LDLIBS = 
EXEC = training
SRC=$(wildcard *.cpp)
OBJ=$(SRC:.cpp=.o)

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)
	 
main.o: init.h data.h training.h test.h tirage.h Straining.h Stest.h
propagation.o: activations.h perte.h
classic.o: propagation.h utilities.h
Sclassic.o: propagation.h 
Sperso.o: propagation.h Sclassic.h
LMs.o: propagation.h scaling.h utilities.h 
perso.o: propagation.h utilities.h classic.h
incremental.o: propagation.h utilities.h
essai.o: propagation.h classic.h utilities.h
training.o: classic.h LMs.h perso.h incremental.h
Straining.o: Sclassic.h Sperso.h
test.o: init.h training.h utilities.h
Stest.o: init.h Straining.h utilities.h
Stirage.o: init.h propagation.h Straining.h 
 
%.o: %.c
	$(CC) -o $@ -c $< $(CXXFLAGS) $(LDLIBS) $(LDFLAGS)

clean:
	rm -f *.o core

mrproper: clean
	rm -f $(EXEC)
