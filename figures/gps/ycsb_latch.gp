# set terminal pngcairo size 800,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体

set terminal pdf size 8,6 enhanced font 'Times-New-Roman,40,Bold'

# set xlabel " " offset 0,-2
set xlabel ""
set ylabel "Throughput(10^6 txn/s)" offset 0,0
set ylabel font 'Times-New-Roman,48,Bold'
set grid ytics linewidth 4
set ytics font 'Times-New-Roman,48,Bold'

set key font "Times-New-Roman,48,Bold"
set key under box center
set key offset 0,-0.25
set key samplen 3 spacing 0.85

set style data histogram
set yrange [0:*]

set style histogram clustered gap 1 title offset 0,0.25
set style fill solid noborder
set boxwidth 0.95

set xtics font 'Times-New-Roman,48' offset 0,0.11 rotate by -30
# set xtics ("tpl\\\_nw" 0, "tpl\\\_wd" 1, "TO" 2, "MVCC" 3, "Silo" 4, "TicToc" 5)
set xtics ("tplnw" 0, "tplwd" 1, "TO" 2, "MVCC" 3, "Silo" 4, "TicToc" 5)

set output "../pdf/ycsb_ro_latch.pdf"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_ro.txt"
plot throu_pth u (($1)/1e6) ti col , '' u (($2)/1e6) ti col

set output "../pdf/ycsb_mc_latch.pdf"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_mc.txt"
plot throu_pth u (($1)/1e6) ti col , '' u (($2)/1e6) ti col

set output "../pdf/ycsb_hc_latch.pdf"  ###输出的文件名
throu_pth = "../mid/ycsb_latch_hc.txt"
plot throu_pth u (($1)/1e6) ti col , '' u (($2)/1e6) ti col