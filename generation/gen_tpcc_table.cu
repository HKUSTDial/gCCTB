#include <db.cuh>
#include <table.cuh>
#include <runtime.cuh>
#include <env.cuh>
#include <tictoc.cuh>
#include <gacco.cuh>
#include <gputx.cuh>
#include <silo.cuh>
#include <to.cuh>
#include <twopl.cuh>
#include <mvcc.cuh>
#include <timestamp.cuh>
#include <tpcc.cuh>
#include <ctime>
#include <set>
#include <algorithm>

using namespace tpcc;

int txcnt, warehouse_cnt;
int isno;
std::string bench_name;
common::DB_CPU db;
common::DB_GPU *db_gpu;
common::ExecInfo *exec_info;
cudaEvent_t start1, mid1, stop1;

int main(int argc, char **argv)
{
    sscanf(argv[1], "%d", &isno);
    sscanf(argv[2], "%d", &warehouse_cnt);
    sscanf(argv[3], "%d", &txcnt);
    // std::cin >> txnum >> warehouse_cnt;

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
    table_warehouse->InitData(warehouse_cnt);
    table_warehouse->SaveData("./tables/warehouse_wh" + std::to_string(warehouse_cnt));
    table_warehouse->SaveIndex(common::SORTED_ARRAY, 0, "./tables/warehouse_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_district->AddColumns(infos_district, gens_district);
    table_district->InitData(warehouse_cnt * DISTRICTS_PER_W);
    table_district->SaveData("./tables/district_wh" + std::to_string(warehouse_cnt));
    table_district->SaveIndex(
        common::SORTED_ARRAY,
        std::vector<int>{0, 1},
        std::vector<int>{1, DISTRICTS_PER_W},
        "./tables/district_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_customer->AddColumns(infos_customer, gens_customer);
    table_customer->InitData(warehouse_cnt * DISTRICTS_PER_W * CUSTOMERS_PER_D);
    table_customer->SaveData("./tables/customer_wh" + std::to_string(warehouse_cnt));
    table_customer->SaveIndex(
        common::SORTED_ARRAY,
        std::vector<int>{0, 1, 2},
        std::vector<int>{1, CUSTOMERS_PER_D, DISTRICTS_PER_W * CUSTOMERS_PER_D},
        "./tables/customer_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_item->AddColumns(infos_item, gens_item);
    table_item->InitData(ITEMS_NUM);
    table_item->SaveData("./tables/item_wh" + std::to_string(warehouse_cnt));
    table_item->SaveIndex(common::SORTED_ARRAY, 0, "./tables/item_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_stock->AddColumns(infos_stock, gens_stock);
    table_stock->InitData(warehouse_cnt * ITEMS_NUM);
    table_stock->SaveData("./tables/stock_wh" + std::to_string(warehouse_cnt));
    table_stock->SaveIndex(common::SORTED_ARRAY,
                           std::vector<int>{0, 1},
                           std::vector<int>{1, ITEMS_NUM}, "./tables/stock_wh" + std::to_string(warehouse_cnt) + ".idx");

    table_history->AddColumns(infos_history, gens_history);
    table_history->InitData(txcnt);
    table_history->SaveData("./tables/history_wh" + std::to_string(warehouse_cnt));

    table_order->AddColumns(infos_order, gens_order);
    table_order->InitData(txcnt);
    table_order->SaveData("./tables/order_wh" + std::to_string(warehouse_cnt));

    table_neworder->AddColumns(infos_neworder, gens_neworder);
    table_neworder->InitData(txcnt);
    table_neworder->SaveData("./tables/neworder_wh" + std::to_string(warehouse_cnt));

    table_orderline->AddColumns(infos_orderline, gens_orderline);
    table_orderline->InitData(txcnt * 15);
    table_orderline->SaveData("./tables/orderline_wh" + std::to_string(warehouse_cnt));

    return 0;
}