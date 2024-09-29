void graph(){
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaGraphCreate(&graph, 0);
    cudaGraphInstantiate(&graph_exec, graph);
    cudaGraphUpload(graph_exec, stream);

    cudaGraphLaunch(graph_exec, stream);
}
