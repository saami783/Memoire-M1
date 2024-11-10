#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

#define MAX_V    1000000

int vc[MAX_V];

int main(int argc, char* argv[])
{
	int vertex_count;
	int edge_count;
	char tempstr1[10];
	char tempstr2[10];
	char line[100];
	int vc_size;
	int i;
	
	if(argc!=4) 
	{
		cout<<"Usage of VCverifer: ./VCverifier <graph file> <solution file> <option parameter>"<<endl;
		cout<<"the last parameter is about output option when the solution is not a vertex cover. It should be either -all or -single, where -all lists all uncovered edges  and -single terminates the progame once it finds an uncovered edge."<<endl;
	}
		

	ifstream graph_file(argv[1]);
	if(graph_file==NULL)  {
		cout<<"cannot find the file"<<argv[1]<<", please make sure you input the correct filename."<<endl;
		return 0;
	}
	graph_file.getline(line,100);
	while (line[0]!='p') graph_file.getline(line,100);
	sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &vertex_count, &edge_count);

	ifstream vc_file(argv[2]);
	if(vc_file==NULL) {
		cout<<"cannot find the file "<<argv[2]<<", please make sure you input the correct filename."<<endl;
		return 0;
	}
	vc_file.getline(line,100);
	while (line[0]!='p') vc_file.getline(line,100);
	sscanf(line, "%s %s %d", tempstr1, tempstr2, &vc_size);
	
	for (i=1;i<=vertex_count;++i) vc[i]=0;

	int tmp;
	for (i=1; i<=vc_size; ++i)
	{
		vc_file>>tmp;
		vc[tmp]=1;
	}
	vc_file.close();
	
	int listall;
	if(strcmp(argv[3],"-all")==0) listall=1;
	else if(strcmp(argv[3],"-single")==0) listall=0;
	else cout<<"the last parameter should be either -all or -single, where -all lists all uncovered edges and -single terminates the progame once it finds an uncovered edge."<<endl;

	int u,v;
	char c;
	int error=0;
	for (i=0; i<edge_count; i++)
	{
		graph_file>>c;
		graph_file>>u;
		graph_file>>v;
		
		if(vc[u]==0 && vc[v]==0) {
			cout<<"uncovered edge: ("<<u<<","<<v<<")"<<endl;
			error++;
			if(listall==0) break;
			}
	}
	graph_file.close();
	
	if(error==0) cout<<"The vertex set provided in "<<argv[2]<<" is a vertex cover."<<endl;
	else {
		cout<<"The vertex set provided in "<<argv[2]<<" is NOT a vertex cover."<<endl;
		if(listall==1) cout<<"there are toally "<<error<<" uncovered edges."<<endl;
	}

	return 0;
}
