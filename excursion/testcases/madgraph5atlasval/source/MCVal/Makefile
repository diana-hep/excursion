# The list of packages used by the macro:
USED_PKGS = xAODRootAccess xAODEgamma xAODTruth

MC15Validation: MC15Validation.o
	`root-config --ld` -o $@ `root-config --libs` \
		-L$(ROOTCOREDIR)/lib `rc get_ldflags $(USED_PKGS)` $^

clean:
	rm -f MC15Validation.o
	rm -f MC15Validation

.SUFFIXES: .C .o

.C.o:
	`root-config --cxx` -c -o $@ `root-config --cflags` \
		-I$(ROOTCOREDIR)/include `rc get_cxxflags $(USED_PKGS)` $<
