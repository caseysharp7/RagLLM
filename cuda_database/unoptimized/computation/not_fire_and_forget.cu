void graph(){
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    create_graph(&graph);
    cudaGraphInstantiate(&graph_exec, graph);
    cudaGraphUpload(graph_exec, stream);

    cudaGraphLaunch(graph_exec, stream);
}
