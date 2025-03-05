#include <db.cuh>
#include <table.cuh>
#include <runtime.cuh>
#include <env.cuh>
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
#include <tpcc.cuh>
#include <ctime>
#include <set>
#include <algorithm>

using namespace tpcc;

int main(int argc, char **argv)
{
    common::Context context;

    int txcnt, warehouse_cnt, valid_txn_bitoffset, warp_cnt, batch_size, debug;
    int isno;
    std::string bench_name;
    std::string cc_type;

    sscanf(argv[1], "%d", &isno);
    sscanf(argv[2], "%d", &warehouse_cnt);
    sscanf(argv[3], "%d", &txcnt);
    cc_type = std::string(argv[4]);
    sscanf(argv[5], "%d", &valid_txn_bitoffset);
    sscanf(argv[6], "%d", &warp_cnt);
    sscanf(argv[7], "%d", &batch_size);
    sscanf(argv[8], "%d", &debug);
    // std::cin >> txnum >> warehouse_cnt;

    std::string name = isno ? "tpcc_no_wh" : "tpcc_pm_wh";
    name += std::to_string(warehouse_cnt);

    common::DB_CPU db;
    common::DB_GPU *db_gpu;

    InitTables(warehouse_cnt, txcnt, db, db_gpu);

    std::cout << "TRY " << argv[4] << "," << valid_txn_bitoffset << "," << warp_cnt << "," << batch_size << "\n";
    size_t db_size = db.GetMemSize();
    // std::cout << "GPUDB SIZE " << db_size << " " << db_size * 1.0f / (1024.0 * 1024.0) << "\n";

    TaskInfo *task_info = nullptr;

    if (isno)
    {
        bench_name = "new_order";
        task_info = NewOrderPre0(warehouse_cnt, txcnt, db);
    }
    else
    {
        bench_name = "payment";
        task_info = PaymentPre0(warehouse_cnt, txcnt, db);
    }

    std::string path("./benchmark/tpcc.cu");
    std::string include_path("./include");
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

    CUdeviceptr cc_device_info = (CUdeviceptr)cc->ToGPU();
    CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());

    common::ExecInfo *exec_info = new common::ExecInfo(name, cc, task_info->txinfo, txcnt, argv[4],
                                                       valid_txn_bitoffset, warp_cnt, batch_size, debug);

    common::RTCProgram *program = new common::RTCProgram(path, include_path, "bench_tpcc", true);
    program->AddNameExpression("payment");
    program->AddNameExpression("new_order");

    std::vector<std::string> str_opts;
    str_opts.push_back(std::string("-I ") + include_path);
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

    ///////////////////////////////////////////////////////

    void *txset_info = ((common::DynamicTransactionSet_CPU *)(task_info->txinfo))->ToGPU();

    void **args = new void *[8];
    args[0] = &db_gpu;
    args[1] = &(task_info->gpu_txs);
    args[2] = &txset_info;
    args[6] = &cc_device_info;
    args[7] = &(task_info->output);

    cudaEvent_t start, stop, start1, mid1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start1);
    cudaEventCreate(&mid1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start);

    float elapsed_time = 0;

    cudaEventRecord(start1);
    cc->Init0();
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&elapsed_time, start1, stop1);
    exec_info->precompute_time = elapsed_time;

    std::vector<size_t> &batches = cc->batches;
    int batch_st = 0;
    int batch_en = 0;
    size_t block_size = warp_cnt << 5;

    for (int i = 0; i < batches.size(); i++)
    {
        cudaEventRecord(start1);
        size_t batched_txnum = batches[i]; // min(batch_size, txnum - i * batch_size);
        // printf("%d\n", batched_txnum);
        if (!batched_txnum)
            continue;
        batch_en = batch_st + batched_txnum;
        size_t thread_num = batched_txnum << valid_txn_bitoffset;
        dim3 num_blocks(thread_num / block_size + (thread_num % block_size == 0 ? 0 : 1), 1, 1);
        dim3 num_threads(thread_num > block_size ? block_size : thread_num, 1, 1);
        args[3] = &i;
        args[4] = &batch_st;
        args[5] = &batch_en;
        cc->Init(i, batch_st);
        cudaEventRecord(mid1);
        program->Call(bench_name, num_blocks, num_threads, args, cc->streams[i]);
        cudaEventRecord(stop1);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&elapsed_time, start1, mid1);
        exec_info->precompute_time += elapsed_time;
        cudaEventElapsedTime(&elapsed_time, mid1, stop1);
        exec_info->processing_time += elapsed_time;
        batch_st = batch_en;
    }
    cc->Sync();

    ///////////////////////////////////////////////////////

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&(exec_info->tot_time), start, stop);
    exec_info->Analyse();
    if (debug)
        exec_info->Check();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(mid1);
    return 0;
}