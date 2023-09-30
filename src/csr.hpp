template <class DataType>
class CSR_Matrix
{
public:
    int _num_vals;
    int _num_col_indices;
    int _num_row_ptrs;
    DataType* _vals;
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
            _vals=new DataType[num_vals];
            _col_indices=new int[num_col_indices];
            _row_ptrs=new int[num_row_ptrs];
            memset(_vals, 0, sizeof(_vals));
        }
    }
    ~CSR_Matrix()
    {
        delete _vals;
        delete _col_indices;
        delete _row_ptrs;
    }
    void setRowPtrsAt
    (
        int i,
        int val
    )
    {
        _row_ptrs[i]=val;
    }
    void setColIndicesAt
    (
        int i,
        int val
    )
    {
        _col_indices[i]=val;
    }
    void setValAt
    (
        int i,
        int j,
        DataType val
    )
    {
        /*for(int itr = _row_ptrs[i]; itr<_row_ptrs[i+1];itr++){
            if(_col_indices[itr]==j){
                _vals[itr]=val;
            }
        }*/
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid;
        while(low<=high){
            mid=low+(high-low)/2;
            if(_col_indices[mid]==j){
                _vals[mid]=val;
                break;
            }
            else if(_col_indices[mid]<j){
                low=mid+1;
            }
            else{
                high=mid-1;
            }
        }        
    }
    double getValAt
    (
        int i,
        int j
    )
    {
        /*for(int itr = _row_ptrs[i]; itr<_row_ptrs[i+1];itr++){
            if(_col_indices[itr]==j){
                return _vals[itr];
            }
        }*/
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid;
        while(low<=high){
            mid=low+(high-low)/2;
            if(_col_indices[mid]==j){
                return _vals[mid];
            }
            else if(_col_indices[mid]<j){
                low=mid+1;
            }
            else{
                high=mid-1;
            }
        }
        return 0;
    }
    void subractValAt
    (
        int i,
        int j,
        DataType val
    )
    {
        /*for(int itr = _row_ptrs[i]; itr<_row_ptrs[i+1];itr++){
            if(_col_indices[itr]==j){
                _vals[itr]-=val;
                break;
            }
        }*/
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid;
        while(low<=high){
            mid=low+(high-low)/2;
            if(_col_indices[mid]==j){
                _vals[mid]-=val;
                break;
            }
            else if(_col_indices[mid]<j){
                low=mid+1;
            }
            else{
                high=mid-1;
            }
        }
    }
    void addValAt
    (
        int i,
        int j,
        DataType val
    )
    {
        /*for(int itr = _row_ptrs[i]; itr<_row_ptrs[i+1];itr++){
            if(_col_indices[itr]==j){
                _vals[itr]+=val;
                break;
            }
        }*/
        
        int low=_row_ptrs[i];
        int high=_row_ptrs[i+1];
        int mid;
        while(low<=high){
            mid=low+(high-low)/2;
            if(_col_indices[mid]==j){
                _vals[mid]+=val;
                break;
            }
            else if(_col_indices[mid]<j){
                low=mid+1;
            }
            else{
                high=mid-1;
            }
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
           std:: cout<<_vals[i]<<" ";
        }std::cout<<std::endl;
    }
};