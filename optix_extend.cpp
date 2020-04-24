#include <torch/extension.h>

#include <iostream>
#include "optix_prime/optix_primepp.h"

class optix_mesh {
  public:
  optix_mesh(torch::Tensor _F, torch::Tensor _V) : F(_F), V(_V) {
    assert(F.size(1) == 3);
    assert(V.size(1) == 3);

    optix_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    model = optix_context->createModel();
    model->setTriangles(
                F.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, F.data<int>(),
                V.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, V.data<float>());
    model->update(RTP_MODEL_HINT_ASYNC);    
  }
  void update_vert(torch::Tensor _V) {
    V = _V;
    assert(V.size(1) == 3);
    model->setTriangles(
                F.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, F.data<int>(),
                V.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, V.data<float>());
    model->update(RTP_MODEL_HINT_ASYNC);
  }
  std::vector<at::Tensor>  intersect(torch::Tensor Ray) {
      assert(Ray.size(1) == 6);
      model->finish();
      auto query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
      auto Hit = torch::empty({Ray.size(0), 2}, torch::TensorOptions().device(torch::kCUDA));
      query->setRays(Ray.size(0),
                     RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION,
                     RTP_BUFFER_TYPE_CUDA_LINEAR,
                     Ray.data<float>());

      query->setHits(Ray.size(0),
                     RTP_BUFFER_FORMAT_HIT_T_TRIID,
                     RTP_BUFFER_TYPE_CUDA_LINEAR,
                     Hit.data_ptr());
      query->execute(0);
      // return Hit;
      std::vector<at::Tensor> T_ID;

      auto T = Hit.as_strided({Ray.size(0)}, {2});
      T_ID.push_back(T);

      torch::Tensor Hit_int32 = torch::from_blob(Hit.data_ptr(), 
        {Ray.size(0), 2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
      auto ID = Hit_int32.as_strided({Ray.size(0)}, {2}, 1);
      T_ID.push_back(ID);
      return T_ID;
  }
  // ~optix_mesh(){
  //   optix_context.
  // }
  torch::Tensor F;
  torch::Tensor V;
  optix::prime::Context optix_context;
  optix::prime::Model model;
};

// std::vector<at::Tensor>  intersect(torch::Tensor F, torch::Tensor V, torch::Tensor Ray) {
//     assert(F.size(1) == 3);
//     assert(V.size(1) == 3);
//     assert(Ray.size(1) == 6);


//     auto optix_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
//     auto model = optix_context->createModel();
//     model->setTriangles(
//                 F.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, F.data<int>(),
//                 V.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, V.data<float>());
//     model->update(RTP_MODEL_HINT_ASYNC);
//     model->finish();
//     auto query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
//     auto Hit = torch::empty({Ray.size(0), 2}, torch::TensorOptions().device(torch::kCUDA));
//     query->setRays(Ray.size(0),
//                    RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION,
//                    RTP_BUFFER_TYPE_CUDA_LINEAR,
//                    Ray.data<float>());
                   
//     query->setHits(Ray.size(0),
//                    RTP_BUFFER_FORMAT_HIT_T_TRIID,
//                    RTP_BUFFER_TYPE_CUDA_LINEAR,
//                    Hit.data_ptr());
//     query->execute(0);
//     // return Hit;
//     std::vector<at::Tensor> T_ID;

//     auto T = Hit.as_strided({Ray.size(0),1}, {2,1});
//     T_ID.push_back(T);

//     torch::Tensor Hit_int32 = torch::from_blob(Hit.data_ptr(), 
//       {Ray.size(0), 2}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt32));
//     auto ID = Hit_int32.as_strided({Ray.size(0),1}, {2,1}, 1);
//     T_ID.push_back(ID);
//     return T_ID;
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("intersect", &intersect, "optix intersect");
  py::class_<optix_mesh>(m, "optix_mesh")
  .def(py::init<torch::Tensor, torch::Tensor>())
  .def("update_vert", &optix_mesh::update_vert)
  .def("intersect", &optix_mesh::intersect);
}