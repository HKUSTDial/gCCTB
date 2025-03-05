# set terminal pngcairo size 800,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体

set terminal pdf size 19,6 enhanced font 'Times-New-Roman,42,Bold'

set output "../pdf/ycsb_latch.pdf"  ###输出的文件名

set multiplot
unset key

set ylabel "Throughput(10^6 txn/s)" offset 1.6,0
set ylabel font 'Times-New-Roman,48,Bold'
set grid ytics linewidth 4
set ytics font 'Times-New-Roman,48,Bold' offset 0.2,0

set style data histogram
set yrange [0:*]

set style histogram clustered gap 1 title offset 0,0.25
set style fill solid noborder
set boxwidth 0.95

set xtics font 'Times-New-Roman,48' offset 0,0.11 rotate by -40
# set xtics ("tpl\\\_nw" 0, "tpl\\\_wd" 1, "TO" 2, "MVCC" 3, "Silo" 4, "TicToc" 5)
set xtics ("tplnw" 0, "tplwd" 1, "TO" 2, "MVCC" 3, "Silo" 4, "TicToc" 5)

set size 0.37,0.9
set origin 0,0.1
set xlabel '(a) YCSB-RO' offset 0,0.5
plot "../mid/ycsb_latch_ro.txt" u (($1)/1e6) ti col , '' u (($2)/1e6) ti col

set origin 0.32,0.1
set xlabel '(b) YCSB-MC'
plot "../mid/ycsb_latch_mc.txt" u (($1)/1e6) ti col , '' u (($2)/1e6) ti col


set key font "Times-New-Roman,48,Bold"
set key at graph -0.35,screen 0.1
set key box
set key horizontal maxrows 1 columns 2
set key samplen 2

set origin 0.64,0.1
set xlabel '(c) YCSB-HC'
plot "../mid/ycsb_latch_hc.txt" u (($1)/1e6) ti col , '' u (($2)/1e6) ti col

unset multiplot