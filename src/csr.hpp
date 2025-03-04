#include <math.h>
class CSR_Matrix
{
private:
    int _tryLastPos_i;
    int _tryLastPos_j;
    int _tryLastPos_mid;
public:
    int _num_vals;
    int _num_col_indices;
    int _num_row_ptrs;
    double* _vals;
    int* _col_indices;
    int* _row_ptrs;
    CSR_Matrix
    (
        int num_vals,
        int num_col_indices,
        int num_row_ptrs
    )
    {
        _num_vals=num_vals;
        _num_col_indices=num_col_indices;
        _num_row_ptrs=num_row_ptrs;
        if(_num_vals==0 && _num_col_indices==0 && _num_row_ptrs==0){
            _vals=nullptr;
            _col_indices=nullptr;
            _row_ptrs=nullptr;
        }
        else{
            _vals=(double*) calloc(num_vals, sizeof(double));
            _col_indices=new int[num_col_indices];
            _row_ptrs=new int[num_row_ptrs];
            if(_row_ptrs==NULL){
                std::cout<<"_row_ptrs=NULL"<<std::endl;
            }
        }
    }
    ~CSR_Matrix()
    {
        delete _vals;
        delete _col_indices;
        delete _row_ptrs;
    }
    int getValIndexAt
    (
        int i,
        int j
    )
    {
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            return mid;
        }
        return -1;
    }
    void setRowPtrsAt
    (
        int i,
        int val
    )
    {
        if(i<_num_row_ptrs){
            _row_ptrs[i]=val;
        }
        else{
            std::cout<<"Error: "<<i<<" >= "<<_num_row_ptrs<<"!"<<std::endl;
        }
    }
    void setColIndicesAt
    (
        int i,
        int val
    )
    {
        if(i<_num_col_indices){
            _col_indices[i]=val;
        }
        else{
            std::cout<<"Error: "<<i<<" >= "<<_num_col_indices<<"!"<<std::endl;
        }
    }
    void setTryLastPos
    (
        int i,
        int j,
        int mid
    )
    {
        _tryLastPos_i=i;
        _tryLastPos_j=j;
        _tryLastPos_mid=mid;
    }
    void resetTryLastPos
    (
        void
    )
    {
        _tryLastPos_i=-1;
        _tryLastPos_j=-1;
        _tryLastPos_mid=-1;
    }
    void setValAt
    (
        int i,
        int j,
        double val
    )
    {
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            _vals[mid]=val;
            return;
        }
    }
    double getValAt
    (
        int i,
        int j
    )
    {
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            return _vals[mid];
        }
        return 0;
    }
    void subractValAt
    (
        int i,
        int j,
        double val
    )
    {
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            _vals[mid]-=val;
        }
    }
    void addValAt
    (
        int i,
        int j,
        double val
    )
    {
        
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            _vals[mid]+=val;
        }
    }
    void divideValAt
    (
        int i,
        int j,
        double val
    )
    {
        
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            _vals[mid]/=val;
        }
    }
    void sqrtValAt
    (
        int i,
        int j
    )
    {
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid=std::lower_bound(_col_indices+low,_col_indices+high,j)-_col_indices;
        if(mid!=high&&mid<_num_col_indices&&_col_indices[mid]==j){
            setTryLastPos(i,j,mid);
            _vals[mid]=sqrt(_vals[mid]);
        }
    }
    void printMatrix(){
        std::cout<<"row_ptrs:";
        for(int i = 0; i < _num_row_ptrs; i++){
            std::cout<<_row_ptrs[i]<<" ";
        }std::cout<<std::endl;
        std::cout<<"col_indices:";
        for(int i = 0; i < _num_col_indices; i++){
            std::cout<<_col_indices[i]<<" ";
        }std::cout<<std::endl;
        std::cout<<"vals:";
        for(int i = 0; i < _num_vals; i++){
           std:: cout<<_vals[i]<<" ";
        }std::cout<<std::endl;
    }
    void printRowPtrs(){
        std::cout<<"row_ptrs:";
        for(int i = 0; i < _num_row_ptrs; i++){
            std::cout<<_row_ptrs[i]<<" ";
        }std::cout<<std::endl;
    }
    void printColIndices(){
        std::cout<<"col_indices:";
        for(int i = 0; i < _num_col_indices; i++){
            std::cout<<_col_indices[i]<<" ";
        }std::cout<<std::endl;
        std::cout<<"vals:";
    }
    void printVals(){
        std::cout<<"vals:";
        for(int i = 0; i < _num_vals; i++){
           std::cout<<_vals[i]<<" ";
        }std::cout<<std::endl;
    }
};

