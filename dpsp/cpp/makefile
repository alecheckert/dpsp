CXX      = g++
CXXFLAGS = -g -std=c++11

TARGET0   = dpsp/cpp/gsdpdiff
TARGET1   = dpsp/cpp/gsdpdiffdefoc

all: $(TARGET0) $(TARGET1)

$(TARGET0): $(TARGET0).cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET0) $(TARGET0).cpp

$(TARGET1): $(TARGET1).cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(TARGET1).cpp

clean:
	$(RM) $(TARGET0)
	$(RM) $(TARGET1)
