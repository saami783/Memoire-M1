#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

#define MAX_V    10000000

int mis[MAX_V];

int main(int argc, char* argv[])
{
	int vertex_count;
	int edge_count;
	char tempstr1[10];
	char tempstr2[10];
	char line[100];
	int mis_size;
	int i;
	
	if(argc!=4) 
	{
		cout<<"Usage of MISverifer: ./MISverifier <graph file> <solution file> <option parameter>"<<endl;
		cout<<"the last parameter is about output option when the solution is not an independent set. It should be either -all or -single, where -all lists all conflict edges  and -single terminates the progame once it finds a conflict edge."<<endl;
	}
		

	ifstream graph_file(argv[1]);
	if(graph_file==NULL)  {
		cout<<"cannot find the file"<<argv[1]<<", please make sure you input the correct filename."<<endl;
		return 0;
	}
	graph_file.getline(line,100);
	while (line[0]!='p') graph_file.getline(line,100);
	sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &vertex_count, &edge_count);

	ifstream mis_file(argv[2]);
	if(mis_file==NULL) {
		cout<<"cannot find the file "<<argv[2]<<", please make sure you input the correct filename."<<endl;
		return 0;
	}
	mis_file.getline(line,100);
	while (line[0]!='p') mis_file.getline(line,100);
	sscanf(line, "%s %s %d", tempstr1, tempstr2, &mis_size);
	
	for (i=1;i<=vertex_count;++i) mis[i]=0;

	int tmp;
	for (i=1; i<=mis_size; ++i)
	{
		mis_file>>tmp;
		mis[tmp]=1;
	}
	mis_file.close();
	
	int listall;
	if(strcmp(argv[3],"-all")==0) listall=1;
	else if(strcmp(argv[3],"-single")==0) listall=0;
	else cout<<"the last parameter should be either -all or -single, where -all lists all conflict edges and -single terminates the progame once it finds a conflict edge."<<endl;

	int u,v;
	char c;
	int error=0;
	for (i=0; i<edge_count; i++)
	{
		graph_file>>c;
		graph_file>>u;
		graph_file>>v;
		
		if(mis[u]==1 && mis[v]==1) {
			cout<<"conflict edge: ("<<u<<","<<v<<")"<<endl;
			error++;
			if(listall==0) break;
			}
	}
	graph_file.close();
	
	if(error==0) cout<<"The vertex set provided in "<<argv[2]<<" is an independent set."<<endl;
	else {
		cout<<"The vertex set provided in "<<argv[2]<<" is NOT an independent set."<<endl;
		if(listall==1) cout<<"there are toally "<<error<<" conflict edges."<<endl;
	}

	return 0;
}
