// Overall Citations: https://www.youtube.com/watch?v=U4ogK0MIzqk&pp=ygUmc2ViYXN0aWFuIGxlZ3VlIGNvZGluZyBhZHZlbnR1cmUgY2hlc3M%3D
// https://www.youtube.com/watch?v=_vqlIPDR2TU&pp=ygUmc2ViYXN0aWFuIGxlZ3VlIGNvZGluZyBhZHZlbnR1cmUgY2hlc3M%3D 
// Links for specific algorithms and strategies are cited throughout.
// Possible Improvements: bitboards, iterative/selective deepening, board symmetry,
// improve legal move detection, better/more efficient evaluation
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <limits>
#include <fstream>
#include <random>
#include <string>
#include <pybind11/pybind11.h> // https://pybind11.readthedocs.io/en/stable/index.html
#include <pybind11/stl.h> // https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
using namespace std;
namespace py=pybind11;

// Constants
#define E 0
#define B 1
#define W 2
static const int SCORES[8][8]={ // https://www.researchgate.net/publication/283046737_Temporal_difference_learning_Othello_agent
    {100, -20, 10, 5, 5, 10, -20, 100}, 
    {-20, -50, -2, -2, -2, -2, -50, -20}, 
    {10, -2, -1, -1, -1, -1, -2, 10}, 
    {5, -2, -1, -1, -1, -1, -2, 5}, 
    {5, -2, -1, -1, -1, -1, -2, 5}, 
    {10, -2, -1, -1, -1, -1, -2, 10}, 
    {-20, -50, -2, -2, -2, -2, -50, -20}, 
    {100, -20, 10, 5, 5, 10, -20, 100}
};

// 8 directions: (dcol, drow)
static const int DX[8]={1, 1, 0, -1, -1, -1, 0, 1};
static const int DY[8]={0, 1, 1, 1, 0, -1, -1, -1};

// Zobrist hash tables
static uint64_t g_zob_b[64];
static uint64_t g_zob_w[64];
static uint64_t g_zob_player_b;
static uint64_t g_zob_player_w;

// Transposition table: https://en.wikipedia.org/wiki/Transposition_table
struct TTEntry
{
    double eval;
    char flag; // 'E'=exact, 'U'=upper bound, 'L'=lower bound
    int depth;
    vector<pair<int, int>> moves; // (col, row)
};
static unordered_map<uint64_t, TTEntry> tt;

// Returns True if the position at (c, r) is on the board.
// inline is used to accelerate small functions.
static inline bool on_board(int c, int r)
{ 
    return c>=0&&c<8&&r>=0&&r<8; 
}
// Returns the opponent of a player.
static inline int opp(int player)
{ 
    return player==B?W:B; 
}

// Returns a vector of pairs of coordinates of pieces to be flipped if the given piece is played at (col, row).
static vector<pair<int, int>> pieces_to_flip(int board[8][8], int player, int col, int row)
{
    vector<pair<int, int>> result;
    if(board[row][col]!=E) return result;
    for(int d=0;d<8;d++)
    {
        int nc=col+DX[d], nr=row+DY[d];
        vector<pair<int, int>> cands;
        while(on_board(nc, nr)&&board[nr][nc]!=E&&board[nr][nc]!=player)
        {
            cands.push_back({nc, nr});
            nc+=DX[d];
            nr+=DY[d];
        }
        if(on_board(nc, nr)&&board[nr][nc]==player&&!cands.empty()) result.insert(result.end(), cands.begin(), cands.end());
    }
    return result;
}

// Returns true if the move at (col, row) is legal for the given player.
static bool is_legal(int board[8][8], int player, int col, int row)
{
    if(board[row][col]!=E) return false;
    for(int d=0;d<8;d++)
    {
        int nc=col+DX[d], nr=row+DY[d];
        int count=0;
        while(on_board(nc, nr)&&board[nr][nc]!=E&&board[nr][nc]!=player)
        {
            count++;
            nc+=DX[d];
            nr+=DY[d];
        }
        if(count>0&&on_board(nc, nr)&&board[nr][nc]==player) return true;
    }
    return false;
}

// Returns the number of legal moves for the given player.
static int legal_moves_count(int board[8][8], int player)
{
    int total=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++) if(is_legal(board, player, c, r)) total++;
    return total;
}

// Returns the evaluation score of the board for the given player.
// Considers number, positional, and mobility advantages.
static double evaluate(int board[8][8], int player)
{
    int num_b=0, num_w=0, score_b=0, score_w=0;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++)
    {
        int p=board[r][c];
        if(p==B)
        { 
            num_b++; 
            score_b+=SCORES[r][c]; 
        }
        else if(p==W)
        { 
            num_w++; 
            score_w+=SCORES[r][c]; 
        }
    }
    int score_adv=score_b-score_w;
    int num_adv=num_b-num_w;
    int mob_b=legal_moves_count(board, B), mob_w=legal_moves_count(board, W);
    
    if(player!=E&&(player==B?mob_b:mob_w)==0)
    {
        if(num_b>num_w) return 100000.0+num_adv;
        if(num_w>num_b) return -100000.0+num_adv;
        return 0.0;
    }
    int mob_adv=mob_b-mob_w;
    
    // Calculate relative weights of scores dependent on game stage
    double total=(double)(num_b+num_w);
    double t=(total-4.0)/60.0;
    if(t<0.0) t=0.0;
    if(t>1.0) t=1.0;
    double w_score=10.0-9.0*t;
    double w_num=-2.0+17.0*t*t;
    double w_mob=15.0-13.0*t;
    return (double)score_adv*w_score+(double)num_adv*w_num+(double)mob_adv*w_mob;
}

// Plays the move if legal while returning PlayResult.
struct PlayResult
{
    bool legal; 
    uint64_t new_hash; 
    vector<pair<int, int>> flipped; 
};
static PlayResult do_play(int board[8][8], int player, int col, int row, uint64_t h)
{
    PlayResult pr;
    pr.flipped=pieces_to_flip(board, player, col, row);
    if(pr.flipped.empty())
    { 
        pr.legal=false; 
        pr.new_hash=h; 
        return pr; 
    }
    board[row][col]=player;
    for(auto& [fc, fr]:pr.flipped) board[fr][fc]=player;
    const uint64_t* my_zob=(player==B)?g_zob_b:g_zob_w;
    const uint64_t* opp_zob=(player==B)?g_zob_w:g_zob_b;
    h^=my_zob[row*8+col];
    for(auto& [fc, fr]:pr.flipped)
    { 
        h^=opp_zob[fr*8+fc]; 
        h^=my_zob[fr*8+fc]; 
    }
    h^=(player==B)?g_zob_player_b:g_zob_player_w;
    h^=(player==B)?g_zob_player_w:g_zob_player_b;
    pr.legal=true; 
    pr.new_hash=h; 
    return pr;
}
// Undoes the move made by do_play.
static void do_unplay(int board[8][8], int player, int col, int row, const vector<pair<int, int>>& flipped)
{
    board[row][col]=E; 
    int op=opp(player);
    for(auto& [fc, fr]:flipped) board[fr][fc]=op;
}

// Minimax with alpha-beta pruning + transposition table
struct MMResult{ 
    double val; 
    vector<pair<int, int>> moves; 
};
struct MoveCandidate
{ 
    double pre_eval; 
    int col, row; 
    uint64_t new_hash; 
    vector<pair<int, int>> flipped; 
};
static MMResult minimax(int board[8][8], int player, int depth, double alpha, double beta, uint64_t curr_hash)
{
    if(depth==0) return {evaluate(board, player), {}};
    
    // Read from transposition tables
    double alpha_orig=alpha, beta_orig=beta;
    auto it=tt.find(curr_hash);
    if(it!=tt.end()&&it->second.depth>=depth)
    {
        TTEntry& e=it->second;
        if(e.flag=='E') return {e.eval, e.moves};
        if(e.flag=='U'&&e.eval<beta) beta=e.eval;
        if(e.flag=='L'&&e.eval>alpha) alpha=e.eval;
        if(beta<=alpha) return {e.eval, e.moves};
    }
    
    // Pre-sort candidates for more efficient pruning
    vector<MoveCandidate> cands; 
    cands.reserve(20);
    for(int r=0;r<8;r++)
    {
        for(int c=0;c<8;c++)
        {
            PlayResult pr=do_play(board, player, c, r, curr_hash);
            if(pr.legal)
            {
                double pre=evaluate(board, opp(player));
                do_unplay(board, player, c, r, pr.flipped);
                cands.push_back({pre, c, r, pr.new_hash, move(pr.flipped)});
            }
        }
    }
    
    if(cands.empty()) return {evaluate(board, player), {}};
    
    bool is_B=(player==B);
    stable_sort(cands.begin(), cands.end(), [is_B](const MoveCandidate& a, const MoveCandidate& b){
        return is_B?(a.pre_eval>b.pre_eval):(a.pre_eval<b.pre_eval);
    });
    
    // Actual minimax search
    vector<pair<int, int>> best_moves;
    double best_eval=is_B?-numeric_limits<double>::infinity():numeric_limits<double>::infinity();
    for(auto& mv:cands)
    {
        board[mv.row][mv.col]=player;
        for(auto& [fc, fr]:mv.flipped) board[fr][fc]=player;
        MMResult child=minimax(board, opp(player), depth-1, alpha, beta, mv.new_hash);
        double curr_eval=child.val;
        board[mv.row][mv.col]=E;
        int op=opp(player);
        for(auto& [fc, fr]:mv.flipped) board[fr][fc]=op;
        if(is_B)
        {
            if(curr_eval>best_eval)
            { 
                best_eval=curr_eval; 
                best_moves={{mv.col, mv.row}}; 
            }
            else if(curr_eval==best_eval) best_moves.push_back({mv.col, mv.row});
            if(curr_eval>alpha) alpha=curr_eval;
        }
        else
        {
            if(curr_eval<best_eval)
            { 
                best_eval=curr_eval; 
                best_moves={{mv.col, mv.row}}; 
            }
            else if(curr_eval==best_eval) best_moves.push_back({mv.col, mv.row});
            if(curr_eval<beta) beta=curr_eval;
        }
        if(beta<=alpha) break; // alpha-beta pruning: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
    }
    
    // Update transposition table
    char flag='E';
    if(best_eval<=alpha_orig) flag='U';
    else if(best_eval>=beta_orig) flag='L';
    tt[curr_hash]={best_eval, flag, depth, best_moves};
    return {best_eval, best_moves};
}

// https://en.wikipedia.org/wiki/Zobrist_hashing
// Initialize random Zobrist hash values
static void init_zobrist_random()
{
    mt19937_64 rng{random_device{}()}; // Mersenne Twister pseudo-random generator of 64-bit numbers
    for(int i=0;i<64;i++)
    {
        g_zob_b[i]=rng();
        g_zob_w[i]=rng();
    }
    g_zob_player_b=rng();
    g_zob_player_w=rng();
}

// Compute Zobrist hash of the board with the given player.
static uint64_t compute_hash(int board[8][8], int player)
{
    uint64_t h=(player==B)?g_zob_player_b:g_zob_player_w;
    for(int r=0;r<8;r++) for(int c=0;c<8;c++)
    {
        if(board[r][c]==B) h^=g_zob_b[r*8+c];
        else if(board[r][c]==W) h^=g_zob_w[r*8+c];
    }
    return h;
}

// Binary file I/O
// Header:  "RVSAI\0" (6 bytes) + FileVersion (12 bytes)
// Zobrist: g_zob_b[64] + g_zob_w[64] + player_b + player_w (1040 bytes)
// TT:      count uint32, then for each entry:
//          key uint64, eval double, flag char, 
//          depth int32, mc int32, mc x (col int32, row int32)
static const char FILE_MAGIC[]="RVSAI"; // File signature of reversi AI
struct FileVersion 
{
    int major;
    int minor;
    int patch;
};
static const FileVersion FILE_VERSION={0, 1, 3}; // Semantic versioning

static bool load_bin(const string& path)
{
    ifstream f(path, ios::binary);
    if(!f) return false;
    char magic[6]={};
    if(!f.read(magic, 6)) return false;
    if(string(magic, 5)!=string(FILE_MAGIC, 5)) return false;
    FileVersion ver;
    if(!f.read((char*)&ver, sizeof(FileVersion))) return false;
    if(ver.major!=FILE_VERSION.major||ver.minor!=FILE_VERSION.minor||ver.patch!=FILE_VERSION.patch) return false;
    
    // Read Zobrist hashes
    if(!f.read((char*)g_zob_b, 512)) return false;
    if(!f.read((char*)g_zob_w, 512)) return false;
    if(!f.read((char*)&g_zob_player_b, 8)) return false;
    if(!f.read((char*)&g_zob_player_w, 8)) return false;
    
    // Read transposition table
    uint32_t count;
    if(!f.read((char*)&count, 4)) return false;
    for(uint32_t i=0;i<count;i++)
    {
        uint64_t key;
        double eval;
        char flag;
        int32_t depth, mc;
        if(!f.read((char*)&key, 8)) return false;
        if(!f.read((char*)&eval, 8)) return false;
        if(!f.read(&flag, 1)) return false;
        if(!f.read((char*)&depth, 4)) return false;
        if(!f.read((char*)&mc, 4)) return false;
        if(mc<0||mc>64) return false;

        TTEntry e;
        e.eval=eval;
        e.flag=flag;
        e.depth=depth;
        e.moves.reserve(mc);
        for(int m=0;m<mc;m++)
        {
            int32_t col, row;
            if(!f.read((char*)&col, 4)) return false;
            if(!f.read((char*)&row, 4)) return false;
            e.moves.push_back({col, row});
        }
        tt[key]=move(e);
    }
    return f.good();
}

static void save_bin(const string& path)
{
    // Write file header and Zobrist hashes
    ofstream f(path, ios::binary);
    if(!f) return;
    f.write(FILE_MAGIC, 6);
    f.write((char*)&FILE_VERSION, sizeof(FileVersion));
    f.write((char*)g_zob_b, 512);
    f.write((char*)g_zob_w, 512);
    f.write((char*)&g_zob_player_b, 8);
    f.write((char*)&g_zob_player_w, 8);
    
    // Write transposition table
    uint32_t count=(uint32_t)tt.size();
    f.write((char*)&count, 4);
    for(auto& [key, e]:tt)
    {
        int32_t mc=(int32_t)min((int)e.moves.size(), 64);
        f.write((char*)&key, 8);
        f.write((char*)&e.eval, 8);
        f.write(&e.flag, 1);
        f.write((char*)&e.depth, 4);
        f.write((char*)&mc, 4);
        for(int m=0;m<mc;m++)
        {
            int32_t col=e.moves[m].first, row=e.moves[m].second;
            f.write((char*)&col, 4);
            f.write((char*)&row, 4);
        }
    }
}

// Python extension module
PYBIND11_MODULE(_reversi_ai, m)
{
    m.def("init", [](const string& path){
        if(!load_bin(path))
        {
            init_zobrist_random();
            tt.clear();
        }
    }, "Load AI data from file (or generate fresh if missing)");
    m.def("save", [](const string& path){ save_bin(path); }, 
          "Save AI data to file");
    m.def("minimax_moves", 
        [](const vector<vector<int>>& board_2d, int player, int depth, 
           double alpha, double beta){
            int board[8][8];
            for(int r=0;r<8;r++) for(int c=0;c<8;c++) board[r][c]=board_2d[r][c];
            MMResult res=minimax(board, player, depth, alpha, beta, compute_hash(board, player));
            py::dict result;
            result["moves"]=res.moves;
            result["val"]=res.val;
            return result;
        }, 
        "Run minimax. Returns {\"moves\": [(col, row), ...], \"val\": score}");
}