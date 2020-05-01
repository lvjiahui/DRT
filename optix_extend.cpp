#include <torch/extension.h>

#include <iostream>
#include "optix_prime/optix_primepp.h"

class optix_mesh {
  public:
  optix_mesh(unsigned cuda_device)  {
    optix_context = optix::prime::Context::create(RTP_CONTEXT_TYPE_CUDA);
    optix_context->setCudaDeviceNumbers(std::vector<unsigned> {cuda_device});
    model = optix_context->createModel();
  }

  void update_mesh(torch::Tensor _F, torch::Tensor _V) {
    F = _F;
    V = _V;
    assert(F.size(1) == 3);
    assert(V.size(1) == 3);
    // std::cout << _F.device() << F.device() << std::endl;
    update(); 
  }

  void update_vert(torch::Tensor _V) {
    V = _V;
    assert(V.size(1) == 3);
    update(); 
  }

  std::vector<at::Tensor>  intersect(torch::Tensor Ray) {
      assert(builded);
      assert(Ray.size(1) == 6);
      model->finish();
      auto query = model->createQuery(RTP_QUERY_TYPE_CLOSEST);
      auto Hit = torch::empty({Ray.size(0), 2}, torch::TensorOptions().device(Ray.device()));
      // std::cout<< Ray.device() << Hit.device() << std::endl;
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
        {Ray.size(0), 2}, torch::TensorOptions().device(Ray.device()).dtype(torch::kInt32));
      auto ID = Hit_int32.as_strided({Ray.size(0)}, {2}, 1);
      T_ID.push_back(ID);
      return T_ID;
  }

private:

  void update() {
    builded = true;
    model->setTriangles(
                F.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, F.data<int>(),
                V.size(0), RTP_BUFFER_TYPE_CUDA_LINEAR, V.data<float>());
    model->update(RTP_MODEL_HINT_ASYNC);      
  }

  bool builded = false;
  torch::Tensor F;
  torch::Tensor V;
  optix::prime::Context optix_context;
  optix::prime::Model model;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<optix_mesh>(m, "optix_mesh")
  .def(py::init<unsigned>())
  .def("update_mesh", &optix_mesh::update_mesh)
  .def("update_vert", &optix_mesh::update_vert)
  .def("intersect", &optix_mesh::intersect);
}