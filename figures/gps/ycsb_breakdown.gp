# set terminal pngcairo size 800,600 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体

set terminal pdf size 8,6 enhanced font 'Times-New-Roman,46,Bold' linewidth 0.5

set key font "Times-New-Roman,46,Bold"
set key under box center 
set key width 0.05
set key horizontal maxrows 3 columns 3
set key offset -1.8,-0.2
set key samplen 2 spacing 0.85

set style data histogram
set style histogram rowstacked
set style fill pattern border -1
set ytics font 'Times-New-Roman,48,Bold'
set xtics font 'Times-New-Roman,46,Bold'
set xtics offset 0,0.12 rotate by -30

set output "../pdf/ycsb_ro_breakdown.pdf"  ###输出的文件名

plot '../mid/ycsb_ro_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"


set output "../pdf/ycsb_mc_breakdown.pdf"  ###输出的文件名

plot '../mid/ycsb_mc_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"

set output "../pdf/ycsb_hc_breakdown.pdf"  ###输出的文件名

plot '../mid/ycsb_hc_breakdown.txt' using 8 t "pre" ,'' using 5 t "manager" , '' using 3 t "wait" , \
    '' using 6 t "index",'' using 2 t "ts" ,'' using 4 t "abort", '' using 7:xtic(1) t "useful"