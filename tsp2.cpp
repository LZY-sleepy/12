#include "gurobi_c++.h"
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>   
#include <sstream>
#include <chrono>
#include <omp.h>
using namespace std;

// 算法参数
const double ALPHA = 1.0;    // 信息素重要程度
const double BETA = 2.0;     // 启发式因子重要程度
const double RHO = 0.1;      // 信息素蒸发系数
const double Q = 100;        // 信息素增加强度
const int MAX_ITER = 100;    // 最大迭代次数
const int ANT_NUM = 50;      // 蚂蚁数量


string itos(int i) {stringstream s; s << i; return s.str(); }

// 蚁群算法类
class AntColony {
private:
    int n;                              
    vector<vector<double>> distance;    
    vector<vector<double>> pheromone;   
    vector<int> bestTour;              
    double bestLength;                  
    mt19937 gen;

    double calculateDistance(const pair<double,double>& a, const pair<double,double>& b) {
        return sqrt(pow(a.first - b.first, 2) + pow(a.second - b.second, 2));
    }

    vector<int> constructSolution() {
        vector<bool> visited(n, false);
        vector<int> tour;
        int current = uniform_int_distribution<>(0, n-1)(gen);
        
        tour.push_back(current);
        visited[current] = true;
        
        while (tour.size() < n) {
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
            
            tour.push_back(next);
            visited[next] = true;
            current = next;
        }
        
        return tour;
    }

    double calculateTourLength(const vector<int>& tour) {
        double length = 0;
        for (size_t i = 0; i < tour.size(); i++) {
            int from = tour[i];
            int to = tour[(i + 1) % tour.size()];
            length += distance[from][to];
        }
        return length;
    }

public:
    AntColony(const vector<pair<double,double>>& coords) : 
        gen(chrono::steady_clock::now().time_since_epoch().count()) {
        n = coords.size();
        distance.resize(n, vector<double>(n));
        pheromone.resize(n, vector<double>(n, 1.0));
        bestLength = numeric_limits<double>::max();
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                distance[i][j] = calculateDistance(coords[i], coords[j]);
            }
        }
    }

    vector<int> solve() {
        cout << "\n============ 蚁群算法优化开始 ============" << endl;
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            vector<vector<int>> antPaths(ANT_NUM);
            vector<double> pathLengths(ANT_NUM);
            
            #pragma omp parallel for
            for (int k = 0; k < ANT_NUM; k++) {
                antPaths[k] = constructSolution();
                pathLengths[k] = calculateTourLength(antPaths[k]);
                
                #pragma omp critical
                {
                    if (pathLengths[k] < bestLength) {
                        bestLength = pathLengths[k];
                        bestTour = antPaths[k];
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
        
        cout << "蚁群算法最终最优解长度: " << bestLength << endl;
        return bestTour;
    }
    
    double getBestLength() const { return bestLength; }
};
  
    // MTZ约束求解器类
    class MTZSolver {
    private:
        GRBEnv* env;
        GRBModel* model;
        GRBVar** x;  // 边变量
        GRBVar* u;   // MTZ辅助变量
        int n;
        vector<pair<double,double>> coords;
    
    public:
        MTZSolver(const vector<pair<double,double>>& coordinates, const vector<int>& initialTour) {
            coords = coordinates;
            n = coords.size();
            
            try {
                env = new GRBEnv();
                model = new GRBModel(*env);
                
                // 创建变量
                x = new GRBVar*[n];
                for (int i = 0; i < n; i++) {
                    x[i] = new GRBVar[n];
                }
                
                u = new GRBVar[n];
                
                // 添加变量和目标函数
                GRBLinExpr obj = 0;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            double dist = sqrt(pow(coords[i].first - coords[j].first, 2) + 
                                            pow(coords[i].second - coords[j].second, 2));
                            x[i][j] = model->addVar(0.0, 1.0, dist, GRB_BINARY);
                            obj += dist * x[i][j];
                        }
                    }
                    u[i] = model->addVar(0.0, n-1, 0.0, GRB_CONTINUOUS);
                }
                
                model->setObjective(obj, GRB_MINIMIZE);
                
                // 添加约束
                // 1. 每个城市进出度为1
                for (int i = 0; i < n; i++) {
                    GRBLinExpr in = 0, out = 0;
                    for (int j = 0; j < n; j++) {
                        if (i != j) {
                            in += x[j][i];
                            out += x[i][j];
                        }
                    }
                    model->addConstr(in == 1);
                    model->addConstr(out == 1);
                }
                
                // 2. MTZ约束
                for (int i = 1; i < n; i++) {
                    for (int j = 1; j < n; j++) {
                        if (i != j) {
                            model->addConstr(u[i] - u[j] + n * x[i][j] <= n - 1);
                        }
                    }
                }
                
                // 设置求解参数
                model->set(GRB_DoubleParam_TimeLimit, 600);
                model->set(GRB_DoubleParam_MIPGap, 0.01);
                model->set(GRB_IntParam_Threads, 0);
                
                // 设置初始解
                for (size_t i = 0; i < initialTour.size() - 1; i++) {
                    x[initialTour[i]][initialTour[i+1]].set(GRB_DoubleAttr_Start, 1.0);
                }
                x[initialTour.back()][initialTour.front()].set(GRB_DoubleAttr_Start, 1.0);
                
            } catch (GRBException& e) {
                cout << "Error code = " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            }
        }
        
        ~MTZSolver() {
            for (int i = 0; i < n; i++) {
                delete[] x[i];
            }
            delete[] x;
            delete[] u;
            delete model;
            delete env;
        }
        
        vector<int> solve() {
            vector<int> tour;
            try {
                cout << "\n============ MTZ优化开始 ============" << endl;
                model->optimize();
                
                if (model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
                    // 重建路径
                    vector<bool> visited(n, false);
                    int current = 0;
                    tour.push_back(current);
                    visited[current] = true;
                    
                    while (tour.size() < n) {
                        for (int j = 0; j < n; j++) {
                            if (!visited[j] && x[current][j].get(GRB_DoubleAttr_X) > 0.5) {
                                tour.push_back(j);
                                visited[j] = true;
                                current = j;
                                break;
                            }
                        }
                    }
                }
            } catch (GRBException& e) {
                cout << "Error code = " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            }
            return tour;
        }
        
        double getObjectiveValue() {
            return model->get(GRB_DoubleAttr_ObjVal);
        }
    };



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
        // 第一阶段：蚁群算法
        cout << "\n============ 第一阶段：蚁群算法 ============" << endl;
        AntColony aco(coords);
        vector<int> aco_tour = aco.solve();
        double aco_length = aco.getBestLength();
        
        // 选择更好的解作为初始解
        vector<int> initial_tour =  aco_tour ;
        double initial_length = aco_length;
        
        // 第三阶段：MTZ精确求解
        cout << "\n============ 第三阶段：MTZ精确求解 ============" << endl;
        MTZSolver mtz(coords, initial_tour);
        vector<int> final_tour = mtz.solve();
        double final_length = mtz.getObjectiveValue();
        
        // 输出总结果
        cout << "\n============ 优化结果总结 ============" << endl;
        cout << "蚁群算法解长度: " << aco_length << endl;
        cout << "选择的初始解长度: " << initial_length << endl;
        cout << "MTZ最优解长度: " << final_length << endl;
        cout << "最终改进比例: " << (initial_length - final_length) / initial_length * 100 << "%" << endl;
        
        cout << "\n最优路径: ";
        for (size_t i = 0; i < final_tour.size(); i++) {
            cout << final_tour[i] << " ";
            if ((i + 1) % 20 == 0) cout << endl;
        }
        cout << endl;
        
    } catch (GRBException& e) {
        cout << "Gurobi错误 " << e.getErrorCode() << ": " << e.getMessage() << endl;
    } catch (const exception& e) {
        cout << "标准错误: " << e.what() << endl;
    } catch (...) {
        cout << "未知错误" << endl;
    }

    return 0;
}