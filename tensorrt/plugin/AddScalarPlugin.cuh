//
// Created by 92571 on 2024/7/31.
//

#ifndef HPC_ADDSCALARPLUGIN_CUH
#define HPC_ADDSCALARPLUGIN_CUH

#include "include/cookbookHelper.cuh"

namespace {
    static const char *PLUGIN_NAME = {"AddScalar"};
    static const char *PLUGIN_VERSION = {"1"};
}

namespace nvinfer1{

    class AddScalarPlugin : public IPluginV2DynamicExt{
    private:
        const std::string name_;
        std::string namespace_;
        struct {
            float scalar;
        } m_;

    public:
        AddScalarPlugin()  = delete;
        AddScalarPlugin(const std::string &name, float scalar);
        AddScalarPlugin(const std::string &name, const void *buffer, size_t length);
        ~AddScalarPlugin();

        const char
    };
}




#endif //HPC_ADDSCALARPLUGIN_CUH
