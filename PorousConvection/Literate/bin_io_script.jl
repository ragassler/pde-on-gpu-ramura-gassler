## Save and load binary files for arrays
using Plots

"""
Save an array A to a binary file named Aname.bin
Aname: String, name of the array (without extension)
A: Array, the array to be saved


Example:
    save_array("myarray", A)

"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

"""
Load an array A from a binary file named Aname.bin
Aname: String, name of the array (without extension)
A: Array, the array to be loaded


Example:
    load_array("myarray", A)

"""
function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

function main()

    A = rand(Float64, 3, 3)
    save_array("myarray", A)

    B = zeros(Float64, 3, 3)
    load_array("myarray", B)

    @assert A == B
    println("Array saved and loaded successfully!")
    return B
end

B = main()
heatmap(B)