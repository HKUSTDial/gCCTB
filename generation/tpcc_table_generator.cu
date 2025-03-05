#ifndef NVRTC_COMPILE

#include <tpcc.cuh>
#include <table.cuh>
#include <generator.cuh>
#include <type.cuh>
#include <vector>

namespace tpcc
{
    common::GeneratorBase *do_nothing_generator = new common::DoNothingGenerator;

    std::vector<common::TypeWithSize> infos_warehouse{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 4),
        // common::TypeWithSize(common::STRING, 81, 3)
    };

    std::vector<common::GeneratorBase *> gens_warehouse{
        new common::SequenceIntegerGenerator(0, 1, 0xffffffff, 4, SEED),
        new common::RandomFloatGenerator(0.0, 0.2, 4, SEED),
        new common::ConstFloatGenerator(300000, 4),
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_district{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 4),
        // common::TypeWithSize(common::STRING, 81, 3)
    };

    std::vector<common::GeneratorBase *> gens_district{
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, DISTRICTS_PER_W},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
            }),
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
            }),
        new common::RandomFloatGenerator(0.0, 0.2, 4, SEED),
        new common::ConstFloatGenerator(30000, 4),
        new common::ConstIntegerGenerator(3001, 4),
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_customer{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        // common::TypeWithSize(common::STRING, 623, 1),
    };

    std::vector<common::GeneratorBase *> gens_customer{
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, CUSTOMERS_PER_D},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
            }),
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, CUSTOMERS_PER_D},
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, DISTRICTS_PER_W},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
            }),
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, CUSTOMERS_PER_D},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, DISTRICTS_PER_W},
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
            }),
        new common::ConstIntegerGenerator(0, 4),
        new common::ConstFloatGenerator(50000, 4),
        new common::RandomFloatGenerator(0.0, 0.5, 4, SEED),
        new common::ConstFloatGenerator(-10, 4),
        new common::ConstFloatGenerator(10, 4),
        new common::ConstIntegerGenerator(1, 4),
        new common::ConstIntegerGenerator(0, 4),
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_item{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 4),
        // common::TypeWithSize(common::STRING, 74, 2),
    };

    std::vector<common::GeneratorBase *> gens_item{
        new common::SequenceIntegerGenerator(0, 1, 0xffffffff, 4, SEED),
        new common::ConstIntegerGenerator(0, 4),
        new common::ConstFloatGenerator(0, 4),
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_stock{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        // common::TypeWithSize(common::STRING, 290, 6),
    };

    std::vector<common::GeneratorBase *> gens_stock{
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, ITEMS_NUM},
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, 0xffffffff},
            }),
        new common::SequenceIntegerGenerator(
            4,
            SEED,
            std::vector<common::SequenceIntegerGenerator::SequenceInfo>{
                common::SequenceIntegerGenerator::SequenceInfo{0, 0, ITEMS_NUM},
                common::SequenceIntegerGenerator::SequenceInfo{0, 1, 0xffffffff},
            }),
        new common::RandomIntegerGenerator(10, 100, 4, SEED),
        new common::ConstIntegerGenerator(0, 4),
        new common::ConstIntegerGenerator(0, 4),
        new common::ConstIntegerGenerator(0, 4),
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_history{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::INT64, 8, 0),
        // common::TypeWithSize(common::STRING, 24, 0),
    };

    std::vector<common::GeneratorBase *> gens_history{
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        // do_nothing_generator
    };

    std::vector<common::TypeWithSize> infos_order{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 4),
        common::TypeWithSize(common::INT64, 8, 0),
    };

    std::vector<common::GeneratorBase *> gens_order{
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
    };

    std::vector<common::TypeWithSize> infos_neworder{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 4),
    };

    std::vector<common::GeneratorBase *> gens_neworder{
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
    };

    std::vector<common::TypeWithSize> infos_orderline{
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::INT64, 8, 0),
        common::TypeWithSize(common::INT32, 4, 0),
        common::TypeWithSize(common::FLOAT32, 4, 0),
        common::TypeWithSize(common::INT32, 4, 4),
    };

    std::vector<common::GeneratorBase *> gens_orderline{
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
        do_nothing_generator,
    };
}

#endif