#include "monocon.hpp"

int main(){
    
    Params params;
    Monocon monocon(params); 
    monocon.buildFromSerializedEngine();
    monocon.get_bindings();


}