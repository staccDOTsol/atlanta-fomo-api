use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use rand::seq::SliceRandom;
use raydium_cp_swap::curve::CurveCalculator;
use serde::{Deserialize, Serialize};
use clap::Parser;
use anyhow::Result;
use solana_sdk::blake3::Hasher;
use solana_sdk::compute_budget::ComputeBudgetInstruction;
use solana_sdk::instruction::Instruction;
use solana_sdk::program_pack::Pack;
use solana_sdk::transaction::{Transaction, TransactionError};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use solana_sdk::pubkey::Pubkey;
use solana_account_decoder::parse_token::UiTokenAmount;

use solana_sdk::signature::{Keypair, Signer};
use solana_client::rpc_client::RpcClient;
use std::str::FromStr;
use configparser::ini::Ini;
use cached::proc_macro::cached;
use cached::SizedCache;
use base64;
use anchor_lang::{AccountDeserialize, AnchorDeserialize};
use anyhow::{format_err};
use arrayref::array_ref;
use raydium_cp_swap::states::PoolState;
use raydium_cp_swap::{curve::constant_product::ConstantProductCurve, states::AmmConfig};
use spl_associated_token_account::get_associated_token_address_with_program_id;

#[cached(
    ty = "SizedCache<Pubkey, Pubkey>",
    create = "{ SizedCache::with_size(1000) }",
    convert = r#"{ account }"#
)]
fn get_account_owner(rpc_client: &RpcClient, account: Pubkey) -> Pubkey {
    rpc_client.get_account(&account)
    .map(|account| account.owner)
    .unwrap_or_else(|_| Pubkey::default())
}
mod instructions;

use instructions::{
    amm_instructions::*,
    rpc::*,
    token_instructions::*,
    utils::*,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ClientConfig {
    http_url: String,
    ws_url: String,
    payer_path: String,
    admin_path: String,
    raydium_cp_program: Pubkey,
    slippage: f64,
}

#[derive(Deserialize)]
struct SwapRequest {
    pool_id: String,
    user_input_token: String,
    amount: u64,
    user_output_token: String,
}

#[derive(Serialize)]
struct SwapResponse {
    success: bool,
    output_amount: u64,
    transaction: String,
}

async fn swap_base_in(req: web::Json<SwapRequest>) -> impl Responder {
    // Parse input parameters
    let pool_id = match Pubkey::from_str(&req.pool_id) {
        Ok(pubkey) => pubkey,
        Err(_) => return HttpResponse::BadRequest().body("Invalid pool_id"),
    };
    let user_input_token = match Pubkey::from_str(&req.user_input_token) {
        Ok(pubkey) => pubkey,
        Err(_) => return HttpResponse::BadRequest().body("Invalid user_input_token"),
    };
    let user_input_amount = req.amount;
    let user_output_token = match Pubkey::from_str(&req.user_output_token) {
        Ok(pubkey) => pubkey,
        Err(_) => return HttpResponse::BadRequest().body("Invalid user_output_token"),
    };
    // Load client config and payer
    let pool_config = load_cfg("config.ini".to_string());
    let payer = read_keypair_file(&pool_config.payer_path);

    // Create RPC client
    let rpc_client = RpcClient::new(pool_config.http_url.clone());

    // Call swap_base_in function
    let swap_result = web::block(move || {
        swap_base_in_logic(
            &rpc_client,
            &pool_config,
            &payer,
            pool_id,
            user_input_token,
            user_input_amount,
            user_output_token,
        )
    }).await;

    match swap_result {
        Ok(Ok((transaction, output_amount))) => HttpResponse::Ok().json(SwapResponse {
            success: true,
            output_amount,
            transaction: base64::encode(bincode::serialize(&transaction).unwrap()),
        }),
        Ok(Err(e)) => HttpResponse::InternalServerError().body(format!("Error: {:?}", e)),
        Err(e) => HttpResponse::InternalServerError().body(format!("Block error: {:?}", e)),
    }
}

#[cached(
    ty = "SizedCache<String, ClientConfig>",
    create = "{ SizedCache::with_size(1) }",
    convert = r#"{ client_config.to_string() }"#
)]
fn load_cfg(client_config: String) -> ClientConfig {
    let mut config = Ini::new();
    let _map = config.load(&client_config).unwrap();
    let http_url = config.get("Global", "http_url").unwrap();
    let ws_url = config.get("Global", "ws_url").unwrap();
    let payer_path = config.get("Global", "payer_path").unwrap();
    let admin_path = config.get("Global", "admin_path").unwrap();
    let raydium_cp_program_str = config.get("Global", "raydium_cp_program").unwrap();
    let raydium_cp_program = Pubkey::from_str(&raydium_cp_program_str).unwrap();
    let slippage = config.getfloat("Global", "slippage").unwrap().unwrap();

    ClientConfig {
        http_url,
        ws_url,
        payer_path,
        admin_path,
        raydium_cp_program,
        slippage,
    }   
}

fn prepare_swap_instruction(
    pool_config: &ClientConfig,
    pool_id: Pubkey,
    pool_state: &PoolState,
    user_input_token: Pubkey,
    user_output_token: Pubkey,
    input_vault: Pubkey,
    output_vault: Pubkey,
    input_token_mint: Pubkey,
    output_token_mint: Pubkey,
    user_input_amount: u64,
    minimum_amount_out: u64,
    instructions: &mut Vec<Instruction>,
    input_token_program: Pubkey,
    output_token_program: Pubkey,
) -> Result<()> {
    let swap_base_in_instr = swap_base_input_instr(
        pool_config,
        pool_id,
        pool_state.amm_config,
        pool_state.observation_key,
        user_input_token,
        user_output_token,
        input_vault,
        output_vault,
        input_token_mint,
        output_token_mint,
        input_token_program,
        output_token_program,
        user_input_amount,
        (minimum_amount_out as f64 * 0.9 ) as u64
    )?;
    for instruction in swap_base_in_instr {
        maybe_add_instruction(instructions, instruction);
    }
    Ok(())
}
fn maybe_add_instruction(instructions: &mut Vec<Instruction>, new_instruction: Instruction) {
    // Create a hash of the instruction
    let mut hasher = Hasher::default();
    hasher.hash(&bincode::serialize(&new_instruction).unwrap());
    let hash = hasher.result();

    // Check if we've seen this instruction before
    if !instructions.iter().any(|instr| {
        let mut instr_hasher = Hasher::default();
        instr_hasher.hash(&bincode::serialize(instr).unwrap());
        instr_hasher.result() == hash
    }) {
        instructions.push(new_instruction);
    }
}

#[cached(
    ty = "SizedCache<String, Arc<Keypair>>",
    create = "{ SizedCache::with_size(10) }",
    convert = r#"{ s.to_string() }"#
)]
fn read_keypair_file(s: &str) -> Arc<Keypair> {
    Arc::new(solana_sdk::signature::read_keypair_file(s).unwrap())
}

use std::cmp::Reverse;
use ordered_float::OrderedFloat;

// Define a structure for the pool edge in the graph
#[derive(Clone, Eq, PartialEq, Debug)]
struct PoolEdge {
    from_token: Pubkey,
    to_token: Pubkey,
    pool_id: Pubkey,
    weight: OrderedFloat<f64>,
    reverse: bool,
    is_lp_token: bool,
    pool_index: usize,
}

// Define a structure for the state in the priority queue
#[derive(Eq, PartialEq, Debug)]
struct State {
    amount: Reverse<OrderedFloat<f64>>, // Max-heap based on amount
    token: Pubkey,
    path: Vec<PoolEdge>, // Keep track of the path
    path_indices: Vec<usize>,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.amount.cmp(&other.amount)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

struct VisitGuard<'a> {
    visited_tokens: &'a mut HashSet<Pubkey>,
    token: Pubkey,
}

impl<'a> Drop for VisitGuard<'a> {
    fn drop(&mut self) {
        self.visited_tokens.remove(&self.token);
    }
}
fn find_best_route(
    rpc_client: &RpcClient,
    pools: &[Pool],
    input_token: Pubkey,
    output_token: Pubkey,
    amount_in: u64,
    discounted_routes: &mut Vec<Vec<usize>>,
    visited_tokens: &mut HashSet<Pubkey>,
  //  alt_manager: &mut AltManager,
) -> Result<Vec<(u64, Vec<PoolEdge>)>> {
    println!("Starting find_best_route function");
    println!("input_token: {:?}", input_token);
    println!("output_token: {:?}", output_token);
    println!("pools length: {:?}", pools.len());

    // Shuffle the pools
    let mut shuffled_pools = pools.to_vec();
    let mut rng = rand::thread_rng();
    shuffled_pools.shuffle(&mut rng);
    let pools = shuffled_pools;

    let mut graph: HashMap<Pubkey, Vec<PoolEdge>> = HashMap::new();
    for (pool_index, pool) in pools.iter().enumerate() {
        let token_a = pool.pool.token_0_mint;
        let token_b = pool.pool.token_1_mint;

        // Add edges for both directions
        let edge_a_to_b = PoolEdge {
            from_token: token_a,
            to_token: token_b,
            pool_id: pool.pubkey,
            weight: OrderedFloat(0.0),
            reverse: false,
            pool_index,
            is_lp_token: false,
        };
        let edge_b_to_a = PoolEdge {
            from_token: token_b,
            to_token: token_a,
            pool_id: pool.pubkey,
            weight: OrderedFloat(0.0),
            reverse: true,
            pool_index,
            is_lp_token: false,
        };
        graph.entry(token_a).or_insert_with(Vec::new).push(edge_a_to_b);
        graph.entry(token_b).or_insert_with(Vec::new).push(edge_b_to_a);
    }

    let mut best_paths = Vec::new();
    let mut queue = BinaryHeap::new();
    queue.push(State {
        amount: Reverse(OrderedFloat(amount_in as f64)),
        token: input_token,
        path: Vec::new(),
        path_indices: Vec::new(),
    });

    while let Some(State { amount, token, path, path_indices }) = queue.pop() {
        if token == output_token {
            best_paths.push((amount.0.into_inner() as u64, path));
            if best_paths.len() >= 10 {
                break;
            }
            continue;
        }

        if path.len() >= 3 {
            continue;
        }

        if let Some(edges) = graph.get(&token) {
            for edge in edges {
                if path_indices.contains(&edge.pool_index) {
                    continue;
                }

                match calculate_swap_output(
                    rpc_client,
                    &pools[edge.pool_index].pool,
                    amount.0.into_inner() as u64,
                    edge.from_token,
                    edge.to_token,
                ) {
                    Ok(output_amount) => {
                        let mut new_path = path.clone();
                        new_path.push(edge.clone());
                        let mut new_path_indices = path_indices.clone();
                        new_path_indices.push(edge.pool_index);
                        queue.push(State {
                            amount: Reverse(OrderedFloat(output_amount as f64)),
                            token: edge.to_token,
                            path: new_path,
                            path_indices: new_path_indices,
                        });
                    },
                    Err(e) => {
                        println!("Error calculating swap output for pool {}: {:?}", edge.pool_index, e);
                    },
                }
            }
        }
    }

    best_paths.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(best_paths)
}


fn calculate_swap_output(
    rpc_client: &RpcClient,
    pool: &PoolState,
    amount_in: u64,
    from_token: Pubkey,
    to_token: Pubkey,
) -> Result<u64> {
    let (reserve_0, reserve_1) = get_pool_reserves(rpc_client, pool);
    if reserve_0 == 0 || reserve_1 == 0 {
        return Err(format_err!("Pool has zero reserves."));
    }

    let (total_input_amount, total_output_amount) = if from_token == pool.token_0_mint {
        (reserve_0, reserve_1)
    } else {
        (reserve_1, reserve_0)
    };

    let amm_config = rpc_client.get_account(&pool.amm_config)?;
    let amm_config_data = amm_config.data.as_slice();
    let amm_config_state: AmmConfig = try_deserialize_unchecked_from_bytes(&amm_config_data)
        .map_err(|e| format_err!("Failed to deserialize AmmConfig: {}", e))?;
    let (input_token_creator_rate, input_token_lp_rate) = if from_token == pool.token_0_mint {
        (amm_config_state.token_0_creator_rate, amm_config_state.token_0_lp_rate)
    } else {
        (amm_config_state.token_1_creator_rate, amm_config_state.token_1_lp_rate)
    };

    let result = CurveCalculator::swap_base_input(
        u128::from(amount_in),
        u128::from(total_input_amount),
        u128::from(total_output_amount),
        input_token_creator_rate,
        input_token_lp_rate,
    ).ok_or(format_err!("Swap calculation failed"))?;

    let amount_out = u64::try_from(result.destination_amount_swapped)
        .map_err(|_| format_err!("Amount out overflow"))?;

    Ok(amount_out)
}
use spl_token_2022::state::Mint;
#[cached(
    ty = "SizedCache<Pubkey, u8>",
    create = "{ SizedCache::with_size(1000) }",
    convert = r#"{ *token_mint }"#
)]
fn get_token_decimals(
    rpc_client: &RpcClient,
    token_mint: &Pubkey,
) -> u8 {

    let account = rpc_client.get_account(token_mint).unwrap_or_default();
    let mint = Mint::unpack(&account.data).unwrap_or_default();
    mint.decimals
}

#[cached(
    ty = "SizedCache<Pubkey, (u64, u64)>",
    create = "{ SizedCache::with_size(1000) }",
    convert = r#"{ pool.token_0_vault }"#,
)]
fn get_pool_reserves(rpc_client: &RpcClient, pool: &PoolState) -> ((u64, u64)) {
    let token_0_vault = rpc_client.get_token_account_balance(&pool.token_0_vault).unwrap_or(UiTokenAmount {
        amount: "0".to_string(),
        ui_amount: None,
        ui_amount_string: "0".to_string(),
        decimals: 0,
    });
    let token_1_vault = rpc_client.get_token_account_balance(&pool.token_1_vault).unwrap_or(UiTokenAmount {
        amount: "0".to_string(),
        ui_amount: None,
        ui_amount_string: "0".to_string(),
        decimals: 0,
    });
    let reserve_a = token_0_vault.amount.parse::<u64>().unwrap_or_default();
    let reserve_b = token_1_vault.amount.parse::<u64>().unwrap_or_default();
    (reserve_a, reserve_b)
}
#[derive(Debug, Clone)]
struct Pool {
    pubkey: Pubkey,
    pool: PoolState,
}
    use anyhow::anyhow;
pub fn try_deserialize_unchecked_from_bytes<T: AccountDeserialize>(data: &[u8]) -> Result<T> {
    T::try_deserialize(&mut data.as_ref())
        .map_err(|e| anyhow!("Failed to deserialize account: {}", e))
}
pub fn try_deserialize_unchecked_from_bytes_zc(input: &[u8]) -> Result<PoolState> {
    if input.is_empty() {
        return Err(anyhow::anyhow!("Input data is empty"));
    }
    let pool_state = unsafe {
        let pool_state_ptr = input[8..].as_ptr() as *const PoolState;
        std::ptr::read_unaligned(pool_state_ptr)
    };
    Ok(pool_state)
}

#[cached(
    ty = "SizedCache<Pubkey, Vec<Pool>>",
    create = "{ SizedCache::with_size(1) }",
    convert = r#"{ *amm_program_id }"#
)]
fn fetch_all_pools(rpc_client: &RpcClient, amm_program_id: &Pubkey) -> Vec<Pool> {
    let mut pools = Vec::new();

    let accounts = rpc_client.get_program_accounts(amm_program_id).unwrap();
    for (pubkey, account) in accounts {
        let pool_data = account.data;
        if pool_data.len() > PoolState::LEN - 64 && pool_data.len() < PoolState::LEN + 64 {
            if let Ok(pool) = try_deserialize_unchecked_from_bytes_zc(&pool_data) {
                pools.push(Pool { pubkey, pool });
            }
        }
    }
    pools
}
fn swap_base_in_logic(
    rpc_client: &RpcClient,
    pool_config: &ClientConfig,
    payer: &Arc<Keypair>,
    pool_id: Pubkey,
    user_input_token: Pubkey,
    user_input_amount: u64,
    user_output_token: Pubkey,
) -> Result<(solana_sdk::transaction::Transaction, u64)> {
    let pool = rpc_client.get_account(&pool_id)?;
    let pool_state = try_deserialize_unchecked_from_bytes_zc(&pool.data)?;
   

    let load_pubkeys = vec![
        pool_state.amm_config,
        pool_state.token_0_vault,
        pool_state.token_1_vault,
        pool_state.token_0_mint,
        pool_state.token_1_mint,
        user_input_token,
    ];
    let rsps = rpc_client.get_multiple_accounts(&load_pubkeys)?;
    let epoch = rpc_client.get_epoch_info()?.epoch;
    let [amm_config_account, token_0_vault_account, token_1_vault_account, token_0_mint_account, token_1_mint_account, user_input_token_account] =
        array_ref![rsps, 0, 6];

    // Fetch all pools
    let pools = fetch_all_pools(rpc_client, &pool_config.raydium_cp_program);
    let token_0_mint = pool_state.token_0_mint;
    let token_1_mint = pool_state.token_1_mint;
    let token_0_vault = pool_state.token_0_vault;
    let token_1_vault = pool_state.token_1_vault;
    // Find the best route
    let mut discounted_routes = Vec::new();
    let mut visited_tokens = HashSet::new();
    let best_routes = find_best_route(
        rpc_client,
        &pools,
        user_input_token,
user_output_token,            user_input_amount,
        &mut discounted_routes,
        &mut visited_tokens,
    )?;

    if best_routes.is_empty() {
        return Err(anyhow::anyhow!("No valid route found"));
    }

    // Use the best route
    let (output_amount, best_route) = &best_routes[0];

    // Create instructions for the best route
    let mut instructions = Vec::new();
    let mut current_input_token = user_input_token;
    let mut current_input_amount = user_input_amount;

    for edge in best_route {
        let pool = pools.iter().find(|p| p.pubkey == edge.pool_id).unwrap();
        
        if edge.is_lp_token {
            if edge.reverse {
                // Withdraw LP tokens
                let withdraw_instructions = withdraw_instr(
                    pool_config,
                    edge.pool_id,
                    pool.pool.token_0_mint,
                    pool.pool.token_1_mint,
                    pool.pool.lp_mint,
                    pool.pool.token_0_vault,
                    pool.pool.token_1_vault,
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.token_0_mint, &get_account_owner(rpc_client, pool.pool.token_0_mint)),
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.token_1_mint, &get_account_owner(rpc_client, pool.pool.token_1_mint)),
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.lp_mint, &get_account_owner(rpc_client, pool.pool.lp_mint)),
                    current_input_amount,
                    0,
                    0,
                )?;
                instructions.extend(withdraw_instructions);
            } else {
                // Deposit to get LP tokens
                let deposit_instructions = deposit_instr(
                    pool_config,
                    edge.pool_id,
                    pool.pool.token_0_mint,
                    pool.pool.token_1_mint,
                    pool.pool.lp_mint,
                    pool.pool.token_0_vault,
                    pool.pool.token_1_vault,
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.token_0_mint, &get_account_owner(rpc_client, pool.pool.token_0_mint)),
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.token_1_mint, &get_account_owner(rpc_client, pool.pool.token_1_mint)),
                    get_associated_token_address_with_program_id(&payer.pubkey(), &pool.pool.lp_mint, &get_account_owner(rpc_client, pool.pool.lp_mint)),
                    0,
                    current_input_amount,
                    0,
                )?;
                instructions.extend(deposit_instructions);
            }
        } else {
            // Regular swap
            let (input_vault, output_vault) = if edge.reverse {
                (pool.pool.token_1_vault, pool.pool.token_0_vault)
            } else {
                (pool.pool.token_0_vault, pool.pool.token_1_vault)
            };

            prepare_swap_instruction(
                pool_config,
                edge.pool_id,
                &pool.pool,
                get_associated_token_address_with_program_id(
                    &payer.pubkey(),
                    &current_input_token,
                    &get_account_owner(rpc_client, current_input_token),
                ),
                get_associated_token_address_with_program_id(
                    &payer.pubkey(),
                    &edge.to_token,
                    &get_account_owner(rpc_client, edge.to_token),
                ),
                input_vault,
                output_vault,
                current_input_token,
                edge.to_token,
                current_input_amount,
                0, // Set minimum output amount
                &mut instructions,
                get_account_owner(rpc_client, current_input_token),
                get_account_owner(rpc_client, edge.to_token),
            )?;
        }

        current_input_token = edge.to_token;
        current_input_amount = calculate_swap_output(
            rpc_client,
            &pool.pool,
            current_input_amount,
            current_input_token,
            edge.to_token,
        )?;
    }

    // Add compute budget instructions
    instructions.insert(0, ComputeBudgetInstruction::set_compute_unit_price(333333));
    instructions.insert(1, ComputeBudgetInstruction::set_compute_unit_limit(1_400_000));

    // Create transaction
    let recent_hash = rpc_client.get_latest_blockhash()?;
    let transaction = solana_sdk::transaction::Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer.pubkey()),
        &[payer.as_ref()],
        recent_hash,
    );
    
    Ok((transaction, *output_amount))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(
                actix_cors::Cors::default()
                    .allow_any_origin()
                    .allow_any_method()
                    .allow_any_header()
            )
            .route("/swap_base_in", web::post().to(swap_base_in))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}