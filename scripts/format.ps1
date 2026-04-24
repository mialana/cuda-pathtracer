Get-ChildItem -Recurse -Path src, stream_compaction `
    -Include *.h, *.cpp, *.inl, *.cu | 
    ForEach-Object { clang-format -i $_.FullName}
