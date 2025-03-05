#include <db.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tpcc.cuh>

#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
#include <slow_to.cuh>
#include <slow_mvcc.cuh>
#include <twopl.cuh>
#include <mvcc.cuh>
#include <timestamp.cuh>

#include <ctime>
#include <set>
#include <algorithm>

using namespace tpcc;

int txcnt, warehouse_cnt, valid_txn_bitoffset, warp_cnt, batch_size, debug;
std::string cc_type;

common::ConcurrencyControlCPUBase *ConstructCC(TaskInfo *task_info, common::DB_CPU &db, common::DB_GPU *&db_gpu)
{
    common::ConcurrencyControlCPUBase *cc = nullptr;
    common::TransactionSet_CPU *txinfo = task_info->txinfo;
    char *gpu_gacco_txs = task_info->gpu_gaccotxs;

    common::TSAllocator_CPU *ts_allocator = new common::NaiveTSAllocator_CPU();
    // common::TSAllocator_CPU *ts_allocator = new common::BatchedTSAllocator_CPU();

    if (cc_type == "tictoc")
        cc = new cc::Tictoc_CPU(&db, txinfo, batch_size);
    else if (cc_type == "silo")
        cc = new cc::Silo_CPU(&db, txinfo, batch_size);
    else if (cc_type == "to")
        cc = new cc::TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slowto")
        cc = new cc::Slow_TO_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "mvcc")
        cc = new cc::MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "slowmvcc")
        cc = new cc::Slow_MVCC_CPU(&db, txinfo, batch_size, ts_allocator);
    else if (cc_type == "tpl_nw")
        cc = new cc::TwoPL_CPU(&db, txinfo, false, batch_size);
    else if (cc_type == "tpl_wd")
        cc = new cc::TwoPL_CPU(&db, txinfo, true, batch_size);
    else if (cc_type == "gacco")
        cc = new cc::Gacco_CPU(&db, db_gpu, txinfo, gpu_gacco_txs, batch_size);
    else if (cc_type == "gputx")
        cc = new cc::GPUTx_CPU(&db, db_gpu, txinfo, gpu_gacco_txs, batch_size);

    size_t cc_size = cc->GetMemSize();
    // std::cout << "CC SIZE " << cc_size << " " << cc_size * 1.0f / (1024 * 1024) << "\n";
    return cc;
}

common::RTCProgram *ConstructProgram(common::DB_CPU &db,
                                     common::ExecInfo *exec_info,
                                     common::TransactionSet_CPU *txinfo,
                                     common::ConcurrencyControlCPUBase *cc)
{
    common::RTCProgram *program = new common::RTCProgram("./benchmark/tpcc.cu", "./include", "bench_tpcc", true);
    program->AddNameExpression("payment");
    program->AddNameExpression("new_order");

    std::vector<std::string> str_opts;
    str_opts.push_back(std::string("-I ./include"));
    str_opts.push_back("-I /usr/include");
    str_opts.push_back("-D NVRTC_COMPILE");
    str_opts.push_back("-arch=compute_89");
    // str_opts.push_back("-lineinfo");
    //  str_opts.push_back("-G");
    str_opts.push_back("-D INDEX_0_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_1_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_2_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_3_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D INDEX_4_TYPE=common::SortedArray_GPU");
    str_opts.push_back("-D THREAD_TX_SHIFT=" + std::to_string(valid_txn_bitoffset));

    exec_info->GetCompileOptions(str_opts);
    txinfo->GetCompileOptions(str_opts);
    cc->GetCompileOptions(str_opts);
    db.GetCompileOptions(str_opts);

    std::vector<const char *> opts;
    for (auto &s : str_opts)
        opts.push_back(s.c_str());

    if (!program->Compile(opts))
        exit(1);
    return program;
}

int main(int argc, char **argv)
{
    common::Context context;

    sscanf(argv[1], "%d", &warehouse_cnt);
    sscanf(argv[2], "%d", &txcnt);
    cc_type = std::string(argv[3]);
    sscanf(argv[4], "%d", &valid_txn_bitoffset);
    sscanf(argv[5], "%d", &warp_cnt);
    sscanf(argv[6], "%d", &batch_size);
    sscanf(argv[7], "%d", &debug);

    int half_txcnt = txcnt >> 1;

    common::DB_CPU db;
    common::DB_GPU *db_gpu;

    InitTables(warehouse_cnt, txcnt, db, db_gpu);

    std::cout << "TRY " << cc_type << "," << valid_txn_bitoffset << "," << warp_cnt << "," << batch_size << "\n";

    TaskInfo *pm_task_info = PaymentPre0(warehouse_cnt, half_txcnt, db);
    TaskInfo *no_task_info = NewOrderPre0(warehouse_cnt, half_txcnt, db);

    common::ConcurrencyControlCPUBase *pm_cc = ConstructCC(pm_task_info, db, db_gpu);
    common::ConcurrencyControlCPUBase *no_cc = ConstructCC(no_task_info, db, db_gpu);

    CUdeviceptr pm_cc_device_info = (CUdeviceptr)pm_cc->ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    CUdeviceptr no_cc_device_info = (CUdeviceptr)no_cc->ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    std::string name = "tpcc_mix" + std::to_string(warehouse_cnt);

    common::ExecInfo *pm_exec_info = new common::ExecInfo(name, pm_cc, pm_task_info->txinfo, half_txcnt, argv[3],
                                                          valid_txn_bitoffset, warp_cnt, batch_size, debug);
    common::ExecInfo *no_exec_info = new common::ExecInfo(name, no_cc, no_task_info->txinfo, half_txcnt, argv[3],
                                                          valid_txn_bitoffset, warp_cnt, batch_size, debug);

    common::RTCProgram *pm_program = ConstructProgram(db, pm_exec_info, pm_task_info->txinfo, pm_cc);
    common::RTCProgram *no_program = ConstructProgram(db, no_exec_info, no_task_info->txinfo, no_cc);

    //////////////////////////////////////////////////////////////
    void *pm_txset_info = nullptr;
    void *no_txset_info = ((common::DynamicTransactionSet_CPU *)(no_task_info->txinfo))->ToGPU();

    void **pm_args = new void *[8];
    pm_args[0] = &db_gpu;
    pm_args[1] = &(pm_task_info->gpu_txs);
    pm_args[2] = &pm_txset_info;
    pm_args[6] = &pm_cc_device_info;
    pm_args[7] = &(pm_task_info->output);

    void **no_args = new void *[8];
    no_args[0] = &db_gpu;
    no_args[1] = &(no_task_info->gpu_txs);
    no_args[2] = &no_txset_info;
    no_args[6] = &no_cc_device_info;
    no_args[7] = &(no_task_info->output);

    cudaEvent_t start, stop, start1, mid1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&mid1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start);

    float elapsed_time = 0;

    cudaEventRecord(start1);
    pm_cc->Init0();
    no_cc->Init0();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsed_time, start1, stop1);
    pm_exec_info->precompute_time = elapsed_time;
    no_exec_info->precompute_time = elapsed_time;

    std::vector<size_t> &pm_batches = pm_cc->batches;
    std::vector<size_t> &no_batches = no_cc->batches;
    int pm_batch_st = 0, no_batch_st = 0;
    int pm_batch_en = 0, no_batch_en = 0;
    size_t block_size = warp_cnt << 5;

    int i;
    for (i = 0; i < min(pm_batches.size(), no_batches.size()); i++)
    {
        cudaEventRecord(start1);
        size_t pm_batched_txnum = pm_batches[i];
        size_t no_batched_txnum = no_batches[i];
        // printf("%d\n", batched_txnum);

        pm_batch_en = pm_batch_st + pm_batched_txnum;
        no_batch_en = no_batch_st + no_batched_txnum;

        size_t pm_thread_num = pm_batched_txnum << valid_txn_bitoffset;
        dim3 pm_num_blocks(pm_thread_num / block_size + (pm_thread_num % block_size > 0), 1, 1);
        dim3 pm_num_threads(pm_thread_num > block_size ? block_size : pm_thread_num, 1, 1);

        size_t no_thread_num = no_batched_txnum << valid_txn_bitoffset;
        dim3 no_num_blocks(no_thread_num / block_size + (no_thread_num % block_size > 0), 1, 1);
        dim3 no_num_threads(no_thread_num > block_size ? block_size : no_thread_num, 1, 1);

        if (pm_batched_txnum)
        {
            pm_args[3] = &i;
            pm_args[4] = &pm_batch_st;
            pm_args[5] = &pm_batch_en;
            pm_cc->Init(i, pm_batch_st);
        }

        if (no_batched_txnum)
        {
            no_args[3] = &i;
            no_args[4] = &no_batch_st;
            no_args[5] = &no_batch_en;
            no_cc->Init(i, no_batch_st);
        }

        cudaEventRecord(mid1);

        if (pm_batched_txnum)
            pm_program->Call("payment", pm_num_blocks, pm_num_threads, pm_args, pm_cc->streams[i]);

        if (no_batched_txnum)
            no_program->Call("new_order", no_num_blocks, no_num_threads, no_args, no_cc->streams[i]);

        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&elapsed_time, start1, mid1);
        pm_exec_info->precompute_time += elapsed_time;
        no_exec_info->precompute_time += elapsed_time;
        cudaEventElapsedTime(&elapsed_time, mid1, stop1);
        pm_exec_info->processing_time += elapsed_time;
        no_exec_info->processing_time += elapsed_time;
        pm_batch_st = pm_batch_en;
        no_batch_st = no_batch_en;
    }
    pm_cc->Sync();
    no_cc->Sync();

    ///////////////////////////////////////////////////////

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(pm_exec_info->tot_time), start, stop);
    no_exec_info->tot_time = pm_exec_info->tot_time;
    pm_exec_info->Analyse();
    no_exec_info->Analyse();
    if (debug)
    {
        pm_exec_info->Check();
        no_exec_info->Check();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(mid1);

    return 0;
}