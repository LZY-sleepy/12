#include "gurobi_c++.h"
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
using namespace std;

const double ALPHA = 1.0;    // 信息素重要程度
const double BETA = 2.0;     // 启发式因子重要程度
const double RHO = 0.1;      // 信息素蒸发系数
const double Q = 100;        // 信息素增加强度
const int MAX_ITER = 100;    // 最大迭代次数
const int ANT_NUM = 50;      // 蚂蚁数量

string itos(int i) {stringstream s; s << i; return s.str(); }
double distance(double* x, double* y, int i, int j);
void findsubtour(int n, double** sol, int* tourlenP, int* tour);

class AntColony {
    private:
        int n;                              
        vector<vector<double>> distance;    
        vector<vector<double>> pheromone;   
        vector<int> bestTour;              
        double bestLength;                  
    
        double calculateDistance(const pair<double,double>& a, const pair<double,double>& b) {
            return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
        }
    
    public:
        AntColony(const vector<pair<double,double>>& coords) {
            n = coords.size();
            distance.resize(n, vector<double>(n));
            pheromone.resize(n, vector<double>(n, 1.0));
            bestLength = numeric_limits<double>::max();
            
            // 初始化距离矩阵
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    distance[i][j] = calculateDistance(coords[i], coords[j]);
                }
            }
        }
    
        vector<int> solve() {
            cout << "\n============ 蚁群算法优化开始 ============" << endl;
            mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
            
            for (int iter = 0; iter < MAX_ITER; iter++) {
                vector<vector<int>> antPaths(ANT_NUM);
                vector<double> pathLengths(ANT_NUM);
                
                #pragma omp parallel for
                for (int k = 0; k < ANT_NUM; k++) {
                    vector<bool> visited(n, false);
                    vector<int> path;
                    int current = uniform_int_distribution<>(0, n-1)(gen);
                    
                    path.push_back(current);
                    visited[current] = true;
                    
                    // 构建路径
                    while (path.size() < n) {
                        vector<double> prob;
                        double total = 0;
                        
                        // 计算概率
                        for (int next = 0; next < n; next++) {
                            if (!visited[next]) {
                                double p = pow(pheromone[current][next], ALPHA) * 
                                         pow(1.0/distance[current][next], BETA);
                                prob.push_back(p);
                                total += p;
                            } else {
                                prob.push_back(0);
                            }
                        }
                        
                        // 轮盘赌选择
                        double r = uniform_real_distribution<>(0, total)(gen);
                        double sum = 0;
                        int next = -1;
                        
                        for (int i = 0; i < n && next == -1; i++) {
                            if (!visited[i]) {
                                sum += prob[i];
                                if (sum >= r) {
                                    next = i;
                                }
                            }
                        }
                        
                        if (next == -1) {
                            for (int i = 0; i < n; i++) {
                                if (!visited[i]) {
                                    next = i;
                                    break;
                                }
                            }
                        }
                        
                        path.push_back(next);
                        visited[next] = true;
                        current = next;
                    }
                    
                    // 计算路径长度
                    double length = 0;
                    for (size_t i = 0; i < path.size(); i++) {
                        int from = path[i];
                        int to = path[(i + 1) % path.size()];
                        length += distance[from][to];
                    }
                    
                    antPaths[k] = path;
                    pathLengths[k] = length;
                    
                    #pragma omp critical
                    {
                        if (length < bestLength) {
                            bestLength = length;
                            bestTour = path;
                            cout << "迭代 " << iter << ": 新的最优解 = " << bestLength << endl;
                        }
                    }
                }
                
                // 更新信息素
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        pheromone[i][j] *= (1.0 - RHO);
                    }
                }
                
                for (int k = 0; k < ANT_NUM; k++) {
                    double delta = Q / pathLengths[k];
                    for (size_t i = 0; i < antPaths[k].size(); i++) {
                        int from = antPaths[k][i];
                        int to = antPaths[k][(i + 1) % antPaths[k].size()];
                        pheromone[from][to] += delta;
                        pheromone[to][from] += delta;
                    }
                }
            }
            
            cout << "最终最优解长度: " << bestLength << endl;
            return bestTour;
        }
        
        double getBestLength() const { return bestLength; }
    };

class subtourelim: public GRBCallback {
public:
    GRBVar** vars;
    int n;
    subtourelim(GRBVar** xvars, int xn) {
        vars = xvars;
        n = xn;
    }
protected:
    void callback() {
        try {
            if (where == GRB_CB_MIPSOL) {
                double** x = new double*[n];
                int* tour = new int[n];
                int i, j, len;
                for (i = 0; i < n; i++)
                    x[i] = getSolution(vars[i], n);

                findsubtour(n, x, &len, tour);

                if (len < n) {
                    GRBLinExpr expr = 0;
                    for (i = 0; i < len; i++)
                        for (j = i + 1; j < len; j++)
                            expr += vars[tour[i]][tour[j]];
                    addLazy(expr <= len - 1);
                }

                for (i = 0; i < n; i++)
                    delete[] x[i];
                delete[] x;
                delete[] tour;
            }
        } catch (GRBException e) {
            cout << "Error number: " << e.getErrorCode() << endl;
            cout << e.getMessage() << endl;
        } catch (...) {
            cout << "Error during callback" << endl;
        }
    }
};

void findsubtour(int n, double** sol, int* tourlenP, int* tour) {
    bool* seen = new bool[n];
    int bestind, bestlen;
    int i, node, len, start;

    for (i = 0; i < n; i++)
        seen[i] = false;

    start = 0;
    bestlen = n + 1;
    bestind = -1;
    node = 0;
    while (start < n) {
        for (node = 0; node < n; node++)
            if (!seen[node])
                break;
        if (node == n)
            break;
        for (len = 0; len < n; len++) {
            tour[start + len] = node;
            seen[node] = true;
            for (i = 0; i < n; i++) {
                if (sol[node][i] > 0.5 && !seen[i]) {
                    node = i;
                    break;
                }
            }
            if (i == n) {
                len++;
                if (len < bestlen) {
                    bestlen = len;
                    bestind = start;
                }
                start += len;
                break;
            }
        }
    }

    for (i = 0; i < bestlen; i++)
        tour[i] = tour[bestind + i];
    *tourlenP = bestlen;

    delete[] seen;
}

double distance(double* x, double* y, int i, int j) {
    double dx = x[i] - x[j];
    double dy = y[i] - y[j];
    return sqrt(dx * dx + dy * dy);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "用法: " << argv[0] << " <tsp文件路径>" << endl;
        return 1;
    }

    string tsp_file = argv[1];
    ifstream infile(tsp_file);
    if (!infile.is_open()) {
        cout << "错误: 无法打开文件 " << tsp_file << endl;
        return 1;
    }

    // 读取TSP文件中的坐标
    vector<pair<double, double>> coords;
    string line;
    while (getline(infile, line)) {
        if (line == "NODE_COORD_SECTION") break;
    }

    while (getline(infile, line)) {
        if (line == "EOF") break;
        stringstream ss(line);
        int index;
        double x, y;
        ss >> index >> x >> y;
        coords.emplace_back(x, y);
    }
    infile.close();

    int n = coords.size();
    cout << "问题规模: " << n << " 个城市" << endl;

    try {
        // 第一阶段：蚁群算法求解
        cout << "\n============ 第一阶段：蚁群算法 ============" << endl;
        AntColony aco(coords);
        vector<int> initial_tour = aco.solve();
        double initial_length = aco.getBestLength();
        
        // 准备Gurobi所需的数据结构
        double* x = new double[n];
        double* y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = coords[i].first;
            y[i] = coords[i].second;
        }

        // 第二阶段：Gurobi精确求解
        cout << "\n============ 第二阶段：Gurobi精确求解 ============" << endl;
        
        // 初始化Gurobi环境和模型
        GRBEnv* env = new GRBEnv();
        GRBModel model = GRBModel(*env);
        
        // 设置求解参数
        model.set(GRB_DoubleParam_TimeLimit, 600);    // 10分钟时限
        model.set(GRB_IntParam_LazyConstraints, 1);   // 启用延迟约束
        model.set(GRB_IntParam_Threads, 0);           // 使用所有可用线程
        model.set(GRB_DoubleParam_MIPGap, 0.01);     // 设置1%的求解精度
        
        // 创建变量
        GRBVar** vars = new GRBVar*[n];
        for (int i = 0; i < n; i++) {
            vars[i] = new GRBVar[n];
            for (int j = 0; j <= i; j++) {
                double dist = distance(x, y, i, j);
                vars[i][j] = model.addVar(0.0, 1.0, dist, 
                    GRB_BINARY, "x_" + itos(i) + "_" + itos(j));
                vars[j][i] = vars[i][j];
            }
        }

        // 添加度数约束
        for (int i = 0; i < n; i++) {
            GRBLinExpr expr = 0;
            for (int j = 0; j < n; j++) {
                expr += vars[i][j];
            }
            model.addConstr(expr == 2, "deg2_" + itos(i));
        }

        // 对角线约束
        for (int i = 0; i < n; i++) {
            vars[i][i].set(GRB_DoubleAttr_UB, 0);
        }

        // 注入蚁群算法的初始解
        cout << "正在设置初始解..." << endl;
        for (size_t i = 0; i < initial_tour.size() - 1; i++) {
            int from = initial_tour[i];
            int to = initial_tour[i+1];
            vars[from][to].set(GRB_DoubleAttr_Start, 1.0);
        }
        vars[initial_tour.back()][initial_tour.front()].set(GRB_DoubleAttr_Start, 1.0);

        // 设置回调函数
        subtourelim cb = subtourelim(vars, n);
        model.setCallback(&cb);

        // 开始优化
        cout << "开始Gurobi优化..." << endl;
        model.optimize();

        // 处理结果
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            cout << "\n============ 优化结果 ============" << endl;
            cout << "蚁群算法初始解长度: " << initial_length << endl;
            cout << "Gurobi最优解长度: " << model.get(GRB_DoubleAttr_ObjVal) << endl;
            cout << "改进比例: " << 
                (initial_length - model.get(GRB_DoubleAttr_ObjVal)) / initial_length * 100 
                << "%" << endl;
            cout << "求解时间: " << model.get(GRB_DoubleAttr_Runtime) << " 秒" << endl;

            // 重建最优路径
            double** sol = new double*[n];
            for (int i = 0; i < n; i++) {
                sol[i] = model.get(GRB_DoubleAttr_X, vars[i], n);
            }

            int* final_tour = new int[n];
            int len;
            findsubtour(n, sol, &len, final_tour);
            
            // 输出最优路径
            cout << "\n最优路径: ";
            for (int i = 0; i < len; i++) {
                cout << final_tour[i] << " ";
                if ((i + 1) % 20 == 0) cout << endl;
            }
            cout << endl;

            // 释放内存
            for (int i = 0; i < n; i++) {
                delete[] sol[i];
            }
            delete[] sol;
            delete[] final_tour;
        } else {
            cout << "\n未找到最优解" << endl;
            cout << "求解状态: " << model.get(GRB_IntAttr_Status) << endl;
        }

        // 清理内存
        for (int i = 0; i < n; i++) {
            delete[] vars[i];
        }
        delete[] vars;
        delete[] x;
        delete[] y;
        delete env;

    } catch (GRBException e) {
        cout << "Gurobi错误 " << e.getErrorCode() << ": " << e.getMessage() << endl;
    } catch (const exception& e) {
        cout << "标准错误: " << e.what() << endl;
    } catch (...) {
        cout << "未知错误" << endl;
    }

    return 0;
}