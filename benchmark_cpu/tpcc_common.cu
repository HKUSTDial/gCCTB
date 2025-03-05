#include <db.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tpcc.cuh>

namespace tpcc
{
    std::vector<common::StructType::StructItem> payment_struct_items{
        common::StructType::StructItem("w_id", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("d_id", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("c_id", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("c_d_id", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("c_w_id", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("h_amount", new common::FloatType(common::FLOAT32), 0),
        common::StructType::StructItem("h_date", new common::IntegerType(common::INT64, false), 0)};

    common::StructType payment_struct_type("TxItem", std::move(payment_struct_items));

    std::vector<common::StructType::StructItem> neworder_struct_items{
        common::StructType::StructItem("read", new common::IntegerType(common::INT32, false), 0),
        common::StructType::StructItem("key1", new common::IntegerType(common::INT32, true), 0),
        common::StructType::StructItem("key2", new common::IntegerType(common::INT32, true), 4)};

    common::StructType neworder_struct_type("TxItem", std::move(neworder_struct_items));

    void InitTables(int warehouse_cnt, int txcnt, common::DB_CPU &db, common::DB_GPU *&db_gpu)
    {
        // Gen Table
        common::Table_CPU *table_warehouse = new common::Table_CPU;
        common::Table_CPU *table_district = new common::Table_CPU;
        common::Table_CPU *table_customer = new common::Table_CPU;
        common::Table_CPU *table_item = new common::Table_CPU;
        common::Table_CPU *table_stock = new common::Table_CPU;
        common::Table_CPU *table_history = new common::Table_CPU;
        common::Table_CPU *table_order = new common::Table_CPU;
        common::Table_CPU *table_neworder = new common::Table_CPU;
        common::Table_CPU *table_orderline = new common::Table_CPU;

        table_warehouse->AddColumns(infos_warehouse, gens_warehouse);
        table_warehouse->LoadData(warehouse_cnt, "./tables/warehouse_wh" + std::to_string(warehouse_cnt));
        // table_warehouse->InitData(warehouse_cnt);
        table_warehouse->LoadIndex(common::SORTED_ARRAY, 0, "./tables/warehouse_wh" + std::to_string(warehouse_cnt) + ".idx");

        table_district->AddColumns(infos_district, gens_district);
        table_district->LoadData(warehouse_cnt * DISTRICTS_PER_W, "./tables/district_wh" + std::to_string(warehouse_cnt));
        // table_district->InitData(warehouse_cnt * DISTRICTS_PER_W);
        table_district->LoadIndex(
            common::SORTED_ARRAY,
            // std::vector<int>{0, 1},
            // std::vector<int>{1, DISTRICTS_PER_W},
            "./tables/district_wh" + std::to_string(warehouse_cnt) + ".idx");

        table_customer->AddColumns(infos_customer, gens_customer);
        table_customer->LoadData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D, "./tables/customer_wh" + std::to_string(warehouse_cnt));
        // table_customer->InitData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D);
        table_customer->LoadIndex(
            common::SORTED_ARRAY,
            // std::vector<int>{0, 1, 2},
            // std::vector<int>{1, CUSTOMERS_PER_D, DISTRICTS_PER_W * CUSTOMERS_PER_D},
            "./tables/customer_wh" + std::to_string(warehouse_cnt) + ".idx");

        table_item->AddColumns(infos_item, gens_item);
        table_item->LoadData(ITEMS_NUM, "./tables/item_wh" + std::to_string(warehouse_cnt));
        // table_item->InitData(ITEMS_NUM);
        table_item->LoadIndex(common::SORTED_ARRAY, 0, "./tables/item_wh" + std::to_string(warehouse_cnt) + ".idx");

        table_stock->AddColumns(infos_stock, gens_stock);
        table_stock->LoadData(warehouse_cnt * ITEMS_NUM, "./tables/stock_wh" + std::to_string(warehouse_cnt));
        // table_stock->InitData(warehouse_cnt * ITEMS_NUM);
        table_stock->LoadIndex(common::SORTED_ARRAY,
                               //    std::vector<int>{0, 1},
                               //    std::vector<int>{1, ITEMS_NUM},
                               "./tables/stock_wh" + std::to_string(warehouse_cnt) + ".idx");

        table_history->AddColumns(infos_history, gens_history);
        table_history->LoadData(txcnt, "./tables/history_wh" + std::to_string(warehouse_cnt));
        // table_history->InitData(txcnt);

        table_order->AddColumns(infos_order, gens_order);
        table_order->LoadData(txcnt, "./tables/order_wh" + std::to_string(warehouse_cnt));
        // table_order->InitData(txcnt);

        table_neworder->AddColumns(infos_neworder, gens_neworder);
        table_neworder->LoadData(txcnt, "./tables/neworder_wh" + std::to_string(warehouse_cnt));
        // table_neworder->InitData(txcnt);

        table_orderline->AddColumns(infos_orderline, gens_orderline);
        table_orderline->LoadData(txcnt * 15, "./tables/orderline_wh" + std::to_string(warehouse_cnt));
        // table_orderline->InitData(txcnt * 15);

        db.AddTable(table_warehouse);
        db.AddTable(table_district);
        db.AddTable(table_customer);
        db.AddTable(table_item);
        db.AddTable(table_stock);
        db.AddTable(table_history);
        db.AddTable(table_order);
        db.AddTable(table_neworder);
        db.AddTable(table_orderline);

        db.Init();
        db_gpu = db.ToGPU();
        CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
    }

    TaskInfo *PaymentPre0(int warehouse_cnt, int txcnt, common::DB_CPU &db)
    {
        common::StaticTransactionSet_CPU *txinfo = new common::StaticTransactionSet_CPU(txcnt, &payment_struct_type, &db);

        gputp_operator::Operator *op1 = new gputp_operator::StructMemberAccessOperator(
            new gputp_operator::VariableOperator("tx"),
            "w_id");
        op1->type = new common::IntegerType(common::INT32, true);

        gputp_operator::Operator *op2 = new gputp_operator::BinOperator(
            gputp_operator::ADD,
            new gputp_operator::BinOperator(
                gputp_operator::MUL,
                new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W),
                new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "w_id")),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "d_id"));

        op2->type = new common::IntegerType(common::INT32, true);

        gputp_operator::Operator *op3 = new gputp_operator::BinOperator(
            gputp_operator::ADD,
            new gputp_operator::BinOperator(
                gputp_operator::ADD,
                new gputp_operator::BinOperator(
                    gputp_operator::MUL,
                    new gputp_operator::BinOperator(
                        gputp_operator::MUL,
                        new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_w_id"),
                        new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W)),
                    new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), CUSTOMERS_PER_D)),
                new gputp_operator::BinOperator(
                    gputp_operator::MUL,
                    new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), CUSTOMERS_PER_D),
                    new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_d_id"))),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "c_id"));

        op3->type = new common::IntegerType(common::INT32, true);
        // txinfo->AddAccessor("warehouser", 0, 0, op1, false);
        txinfo->AddAccessor("warehousew", 0, 0, op1, true);
        // txinfo->AddAccessor("districtr", 1, 0, op2, false);
        txinfo->AddAccessor("districtw", 1, 0, op2, true);
        // txinfo->AddAccessor("customerr", 2, 0, op3, false);
        txinfo->AddAccessor("customerw", 2, 0, op3, true);

        PaymentTx *txdata = new PaymentTx[txcnt];
        std::string file_name = std::string("./dataset/tpcc_pm_wh") + std::to_string(warehouse_cnt) + ".txs";
        FILE *file = fopen(file_name.c_str(), "rb");
        fread(txdata, sizeof(PaymentTx), txcnt, file);
        fclose(file);

        size_t tx_size = txcnt * sizeof(PaymentTx);
        char *gpu_txs;
        cudaMalloc(&gpu_txs, tx_size);
        cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

        CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
        delete[] txdata;

        printf("---------------------------------- Payment\n");

        void *output;
        cudaMalloc(&output, sizeof(PaymentOutput) * txcnt);

        return new TaskInfo(txinfo, gpu_txs, gpu_txs, output);
    }

    TaskInfo *NewOrderPre0(int warehouse_cnt, int txcnt, common::DB_CPU &db)
    {
        std::vector<int> rcnts(txcnt);
        std::vector<int> wcnts(txcnt);
        std::vector<int> detailed_opcnts(txcnt * 2);

        NewOrderTx *txdata = new NewOrderTx[txcnt];
        std::string file_name = std::string("./dataset/tpcc_no_wh") + std::to_string(warehouse_cnt) + ".txs";
        FILE *file = fopen(file_name.c_str(), "rb");
        fread(txdata, sizeof(NewOrderTx), txcnt, file);
        fclose(file);

        char *gpu_txs;
        size_t tx_size = txcnt * sizeof(NewOrderTx);
        cudaMalloc(&gpu_txs, tx_size);
        cudaMemcpy(gpu_txs, txdata, tx_size, cudaMemcpyHostToDevice);

        std::vector<NewOrderReq> gacco_txdata;
        for (int i = 0; i < txcnt; i++)
        {
            // gacco_txdata.push_back(NewOrderReq(1, txdata[i].w_id, txdata[i].d_id));
            gacco_txdata.push_back(NewOrderReq(0, txdata[i].w_id, txdata[i].d_id));
            for (int j = 0; j < txdata[i].ol_cnt; j++)
            {
                // gacco_txdata.push_back(NewOrderReq(1, txdata[i].supply_w_ids[j], txdata[i].i_ids[j]));
                gacco_txdata.push_back(NewOrderReq(0, txdata[i].supply_w_ids[j], txdata[i].i_ids[j]));
            }
            rcnts[i] = 0;
            wcnts[i] = txdata[i].ol_cnt + 1;
            detailed_opcnts[i * 2] = 1;
            detailed_opcnts[i * 2 + 1] = txdata[i].ol_cnt;
        }

        char *gpu_gaccotxs;
        cudaMalloc(&gpu_gaccotxs, sizeof(NewOrderReq) * gacco_txdata.size());
        cudaMemcpy(gpu_gaccotxs, gacco_txdata.data(), sizeof(NewOrderReq) * gacco_txdata.size(), cudaMemcpyHostToDevice);

        CUDA_DRIVER_SAFE_CALL(cuCtxSynchronize());
        delete[] txdata;

        printf("-------------------------- NEW ORDER\n");

        common::DynamicTransactionSet_CPU *txinfo = new common::DynamicTransactionSet_CPU(
            txcnt,
            2,
            &neworder_struct_type,
            std::move(rcnts),
            std::move(wcnts),
            std::move(detailed_opcnts),
            &db);

        gputp_operator::Operator *op1 = new gputp_operator::BinOperator(
            gputp_operator::ADD,
            new gputp_operator::BinOperator(
                gputp_operator::MUL,
                new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), DISTRICTS_PER_W),
                new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key1")),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key2"));
        op1->type = new common::IntegerType(common::INT32, true);

        gputp_operator::Operator *op2 = new gputp_operator::BinOperator(
            gputp_operator::ADD,
            new gputp_operator::BinOperator(
                gputp_operator::MUL,
                new gputp_operator::IntegerConstantOperator(new common::IntegerType(common::INT32, true), ITEMS_NUM),
                new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key1")),
            new gputp_operator::StructMemberAccessOperator(new gputp_operator::VariableOperator("tx"), "key2"));
        op2->type = new common::IntegerType(common::INT32, true);

        // txinfo->AddAccessor("district", 1, 0, op1, false);
        // txinfo->AddAccessor("stock", 4, 0, op2, false);
        txinfo->AddAccessor("district", 1, 0, op1, true);
        txinfo->AddAccessor("stock", 4, 0, op2, true);

        void *output;
        cudaMalloc(&output, sizeof(NewOrderOutput) * txcnt);

        return new TaskInfo(txinfo, gpu_txs, gpu_gaccotxs, output);
    }
}