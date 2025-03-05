# set terminal pngcairo size 800,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体

set terminal pdf size 19,6 enhanced font 'Times-New-Roman,42,Bold' linewidth 0.5

set style data histogram
set style histogram rowstacked
set style fill pattern border -1
set ytics font 'Times-New-Roman,48,Bold'
set xtics font 'Times-New-Roman,48,Bold'
set xtics offset 0,0.12 rotate by -40

set output "../pdf/ycsb_breakdown.pdf"  ###输出的文件名

set multiplot

unset key

set size 0.37,0.9
set origin 0,0.1
set xlabel '(a) YCSB-RO' offset 0,0.2
plot '../mid/ycsb_ro_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"

set origin 0.32,0.1
set xlabel '(b) YCSB-MC'
plot '../mid/ycsb_mc_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"

set key font "Times-New-Roman,48,Bold"
set key at graph 1.05,screen 0.1
set key box
set key horizontal maxrows 1 columns 7
set key samplen 2
#set key width 0.03

set origin 0.64,0.1
set xlabel '(c) YCSB-HC'
plot '../mid/ycsb_hc_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"

unset multiplot