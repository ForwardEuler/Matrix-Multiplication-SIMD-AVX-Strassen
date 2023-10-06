//
// Created by Semigroup on 2023/9/21.
//

#ifndef M_MEM_POOL_H
#define M_MEM_POOL_H

#include <cstdlib>
#include <vector>

typedef char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char byte;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

/**
 * @brief WARNING: this class is not thread safe
 */
class memPool
{
    u64 curr_size;
    byte* stack_ptr;
    std::vector<byte*> initptr;
    u64 block_size;

public:
    /**
     * @brief WARNING: this class is not thread safe
     * @param size size of each block in memPool in bytes, pool will automatically expand by this size if full
     */
    explicit memPool(u64 size): block_size(size)
    {
        byte* tmp = (byte*)malloc(size);
        stack_ptr = tmp;
        initptr.push_back(tmp);
        curr_size = 0;
    }

    ~memPool()
    {
        memfree();
    }

    /**
     * @brief allocate memory from pool, pool will automatically expand if full
     * @param size size of memory in bytes to allocate
     * @return void* pointer to the allocated memory (like malloc)
     */
    void* memloc(u64 size)
    {
        curr_size += size;
        block_size = size > block_size ? size : block_size;
        if (curr_size >= block_size)
        {
            byte* tmp = (byte*)malloc(block_size);
            initptr.push_back(tmp);
            curr_size = size;
            stack_ptr = tmp + size;
            return tmp;
        }
        byte* ret_ptr = stack_ptr;
        stack_ptr += size;
        return (void*)ret_ptr;
    }

    /**
     * @brief free all memory in memPool
     */
    void memfree()
    {
        for(auto& a : initptr)
        {
            free(a);
        }
        initptr.clear();
        curr_size = 0;
    }
};
#endif
