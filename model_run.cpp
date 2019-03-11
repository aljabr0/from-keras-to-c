#include "tensorflow/c/c_api.h"
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <vector>
#include <assert.h>
#include <string.h>
#include <fstream>

static TF_Buffer *read_tf_buffer_from_file(const char* file);

class CStatus{
public:
    TF_Status *ptr;
    CStatus(){
        ptr = TF_NewStatus();
    }

    void dump_error()const{
        std::cerr << "TF status error: " << TF_Message(ptr) << std::endl;
    }

    inline bool failure()const{
        return TF_GetCode(ptr) != TF_OK;
    }

    ~CStatus(){
        if(ptr)TF_DeleteStatus(ptr);
    }
};

namespace detail {
    template<class T>
    class TFObjDeallocator;

    template<>
    struct TFObjDeallocator<TF_Status> {
        static void run(TF_Status *obj) { TF_DeleteStatus(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_Graph> {
        static void run(TF_Graph *obj) { TF_DeleteGraph(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_Tensor> {
        static void run(TF_Tensor *obj) { TF_DeleteTensor(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_SessionOptions> {
        static void run(TF_SessionOptions *obj) { TF_DeleteSessionOptions(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_Buffer> {
        static void run(TF_Buffer *obj) { TF_DeleteBuffer(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_ImportGraphDefOptions> {
        static void run(TF_ImportGraphDefOptions *obj) { TF_DeleteImportGraphDefOptions(obj); }
    };

    template<>
    struct TFObjDeallocator<TF_Session> {
        static void run(TF_Session *obj) {
            CStatus status;
            TF_DeleteSession(obj, status.ptr);
            if (status.failure()) {
                status.dump_error();
            }
        }
    };
}

template<class T> struct TFObjDeleter{
    void operator()(T* ptr) const{
        detail::TFObjDeallocator<T>::run(ptr);
    }
};

template<class T> struct TFObjMeta{
    typedef std::unique_ptr<T, TFObjDeleter<T>> UniquePtr;
};

template<class T> typename TFObjMeta<T>::UniquePtr tf_obj_unique_ptr(T *obj){
    typename TFObjMeta<T>::UniquePtr ptr(obj);
    return ptr;
}

class MySession{
public:
    typename TFObjMeta<TF_Graph>::UniquePtr graph;
    typename TFObjMeta<TF_Session>::UniquePtr session;

    TF_Input inputs;
    TF_Output outputs;
};

MySession *my_model_load(const char *filename, const char *input_name, const char *output_name){
    CStatus status;

    auto graph=tf_obj_unique_ptr(TF_NewGraph());
    {
        // Load a protobuf containing a GraphDef
        auto graph_def=tf_obj_unique_ptr(read_tf_buffer_from_file(filename));
        if(!graph_def){
            return nullptr;
        }

        auto graph_opts=tf_obj_unique_ptr(TF_NewImportGraphDefOptions());
        TF_GraphImportGraphDef(graph.get(), graph_def.get(), graph_opts.get(), status.ptr);
    }

    if(status.failure()){
        status.dump_error();
        return nullptr;
    }

    auto input_op = TF_GraphOperationByName(graph.get(), input_name);
    auto output_op = TF_GraphOperationByName(graph.get(), output_name);
    if(!input_op || !output_op){
        return nullptr;
    }

    auto session = std::make_unique<MySession>();
    {
        auto opts = tf_obj_unique_ptr(TF_NewSessionOptions());
        session->session = tf_obj_unique_ptr(TF_NewSession(graph.get(), opts.get(), status.ptr));
    }

    if(status.failure()){
        return nullptr;
    }
    assert(session);

    graph.swap(session->graph);
    session->inputs = {input_op, 0};
    session->outputs = {output_op, 0};

    return session.release();
}

template<class T> static void free_cpp_array(void* data, size_t length){
    delete []((T *)data);
}
static void dummy_deallocator(void* data, size_t length, void* arg){}

static TF_Buffer* read_tf_buffer_from_file(const char* file) {
    std::ifstream t(file);
    t.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    t.seekg(0, std::ios::end);
    size_t size = t.tellg();
    auto data = std::make_unique<char>(size);
    t.seekg(0);
    t.read(data.get(), size);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data.release();
    buf->length = size;
    buf->data_deallocator = free_cpp_array<char>;
    return buf;
}

int main(){
    auto session = std::unique_ptr<MySession>(my_model_load("/tmp/frozen_model.pb", "", ""));

    const char *str="0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000"
                    "0000000000000000000000000000";




    return 0;
}
