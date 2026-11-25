extern "C" __attribute__((global)) void vector_add(const float *a, const float *b,
                                                   float *output, unsigned int count)
{
    // Code object V4 exposes workgroup_size_x at byte 4 of the AQL dispatch packet.
    const auto *dispatch = (const __attribute__((address_space(4))) unsigned short *)
        __builtin_amdgcn_dispatch_ptr();
    unsigned int index = __builtin_amdgcn_workgroup_id_x() * dispatch[2] +
                         __builtin_amdgcn_workitem_id_x();
    if (index < count)
        output[index] = a[index] + b[index];
}
