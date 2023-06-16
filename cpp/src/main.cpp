#include "monocon.hpp"

int main(){
    
    Params params;
    Monocon monocon(params); 
    // monocon.build();
    monocon.buildFromSerializedEngine();
    monocon.get_bindings();


}