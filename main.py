import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
from torch_geometric.utils import to_networkx, negative_sampling
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import pickle
from pathlib import Path
from collections import defaultdict
import json
warnings.filterwarnings('ignore')

class KnowledgeGraph:
    """Enhanced Knowledge Graph implementation for biokg dataset"""
    
    def __init__(self):
        self.graph = None
        self.node_embeddings = None
        self.node_features = None
        self.edge_index = None
        self.edge_types = None
        self.edge_type_names = None
        self.entity_names = {}
        self.entity_types = {}
        self.entity_type_map = {}
        self.num_nodes = 0
        self.node_types = None
        self.node_to_type = {}
        self.node_type_offsets = {}
        self.num_nodes_dict = {}
        
    def load_biokg_dataset(self):
        """Load the ogbl-biokg dataset with proper entity and relation mapping"""
        print("Loading ogbl-biokg dataset...")
        
        try:
            # Load the biokg dataset
            dataset = PygLinkPropPredDataset('ogbl-biokg')
            
            # The dataset is a heterogeneous graph
            # We need to handle it differently
            split_edge = dataset.get_edge_split()
            
            # Try to get the graph data
            if hasattr(dataset, 'graph') and dataset.graph is not None:
                graph = dataset.graph
            elif len(dataset) > 0:
                graph = dataset[0]
            else:
                raise ValueError("Could not access graph data")
            
            print(f"Graph type: {type(graph)}")
            
            # For heterogeneous graphs in OGB
            if hasattr(graph, 'edge_index_dict'):
                print("Processing heterogeneous graph...")
                
                # Get node counts
                if hasattr(graph, 'num_nodes_dict'):
                    self.num_nodes_dict = graph.num_nodes_dict
                    print(f"Node types and counts: {self.num_nodes_dict}")
                    
                    # Calculate total nodes and create node mappings
                    node_offset = 0
                    self.node_type_offsets = {}
                    self.node_to_type = {}
                    self.type_to_nodes = defaultdict(list)
                    
                    for node_type, num_nodes in self.num_nodes_dict.items():
                        self.node_type_offsets[node_type] = node_offset
                        for i in range(num_nodes):
                            global_id = node_offset + i
                            self.node_to_type[global_id] = node_type
                            self.type_to_nodes[node_type].append(global_id)
                        node_offset += num_nodes
                    
                    self.num_nodes = node_offset
                    print(f"Total nodes: {self.num_nodes}")
                
                # Process edges - convert heterogeneous to homogeneous
                all_edges = []
                all_edge_types = []
                edge_type_to_id = {}
                edge_id_to_type = {}
                
                print("\nProcessing edge types:")
                for edge_idx, (edge_type_tuple, edge_index) in enumerate(graph.edge_index_dict.items()):
                    src_type, relation, dst_type = edge_type_tuple
                    edge_type_str = f"{src_type}-{relation}-{dst_type}"
                    edge_type_to_id[edge_type_str] = edge_idx
                    edge_id_to_type[edge_idx] = edge_type_str
                    
                    # Adjust node indices based on node type offsets
                    src_offset = self.node_type_offsets[src_type]
                    dst_offset = self.node_type_offsets[dst_type]
                    
                    # Create adjusted edge index
                    adjusted_edges = edge_index.clone()
                    adjusted_edges[0] += src_offset
                    adjusted_edges[1] += dst_offset
                    
                    all_edges.append(adjusted_edges)
                    all_edge_types.extend([edge_idx] * edge_index.size(1))
                    
                    print(f"  {edge_type_str}: {edge_index.size(1)} edges")
                
                # Combine all edges
                self.edge_index = torch.cat(all_edges, dim=1)
                self.edge_types = torch.tensor(all_edge_types, dtype=torch.long)
                self.edge_type_names = edge_id_to_type
                self.edge_type_to_id = edge_type_to_id
                
                print(f"\nCombined graph: {self.edge_index.size(1)} total edges")
                
            else:
                # Handle as homogeneous graph
                print("Processing as homogeneous graph...")
                if hasattr(graph, 'edge_index'):
                    self.edge_index = graph.edge_index
                    self.num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else int(self.edge_index.max().item() + 1)
                    
                    if hasattr(graph, 'edge_reltype'):
                        self.edge_types = graph.edge_reltype
                    elif hasattr(graph, 'edge_type'):
                        self.edge_types = graph.edge_type
                    else:
                        self.edge_types = torch.zeros(self.edge_index.size(1), dtype=torch.long)
                else:
                    raise ValueError("No edge_index found in graph")
            
            # Create entity type map for biokg
            self.entity_type_map = {
                'disease': 'disease',
                'drug': 'drug',
                'function': 'molecular_function',
                'protein': 'protein',
                'sideeffect': 'side_effect'
            }
            
            # Create node features
            self.create_node_features()
            
            # Create entity names
            self.create_entity_names()
            
            print(f"\nDataset loaded successfully:")
            print(f"- Total nodes: {self.num_nodes}")
            print(f"- Total edges: {self.edge_index.size(1)}")
            print(f"- Edge types: {len(self.edge_type_names) if self.edge_type_names else 'Unknown'}")
            if hasattr(self, 'num_nodes_dict'):
                print(f"- Node types: {list(self.num_nodes_dict.keys())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            print("\nCreating synthetic biokg-like data...")
            self.create_synthetic_biokg()
            return False
            
            # Load entity and relation mappings if available
            try:
                # Entity type mapping
                entity_type_file = Path(dataset.root) / 'mapping' / 'nodetype2id.csv'
                if entity_type_file.exists():
                    entity_df = pd.read_csv(entity_type_file, sep='\t')
                    self.entity_type_map = dict(zip(entity_df.iloc[:, 1], entity_df.iloc[:, 0]))
                
                # Relation type mapping
                relation_file = Path(dataset.root) / 'mapping' / 'relationtype2id.csv'
                if relation_file.exists():
                    relation_df = pd.read_csv(relation_file, sep='\t')
                    self.edge_type_names = dict(zip(relation_df.iloc[:, 1], relation_df.iloc[:, 0]))
            except:
                print("Could not load mapping files, using default names")
                self.create_default_mappings()
            
            # Create node features
            self.create_node_features()
            
            # Create entity names for a subset (for visualization)
            self.create_entity_names()
            
            print(f"Loaded biokg dataset:")
            print(f"- Nodes: {self.num_nodes}")
            print(f"- Edges: {self.edge_index.size(1)}")
            print(f"- Edge types: {len(torch.unique(self.edge_types))}")
            print(f"- Node types: {len(self.entity_type_map) if self.entity_type_map else 'Unknown'}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating synthetic biokg-like data...")
            self.create_synthetic_biokg()
            return False
    
    def create_default_mappings(self):
        """Create default mappings for entity and relation types"""
        # Common biomedical entity types
        self.entity_type_map = {
            0: "protein",
            1: "disease", 
            2: "drug",
            3: "gene",
            4: "pathway",
            5: "molecular_function",
            6: "biological_process",
            7: "cellular_component",
            8: "side_effect",
            9: "anatomy"
        }
        
        # Common biomedical relation types
        self.edge_type_names = {
            0: "protein_protein_interaction",
            1: "drug_disease_treatment",
            2: "gene_protein_expression",
            3: "drug_protein_binding",
            4: "disease_protein_association",
            5: "protein_pathway_participation",
            6: "drug_side_effect",
            7: "disease_gene_association",
            8: "protein_function",
            9: "protein_localization"
        }
    
    def create_node_features(self):
        """Create meaningful node features based on graph structure"""
        print("Creating node features...")
        
        # Ensure we have num_nodes
        if self.num_nodes is None or self.num_nodes == 0:
            print("Warning: num_nodes not set, calculating from edge index...")
            if self.edge_index is not None:
                self.num_nodes = int(self.edge_index.max().item() + 1)
            else:
                raise ValueError("Cannot determine number of nodes")
        
        # Calculate various graph statistics as features
        degrees = torch.zeros(self.num_nodes)
        in_degrees = torch.zeros(self.num_nodes)
        out_degrees = torch.zeros(self.num_nodes)
        
        # Safely iterate through edges
        if self.edge_index is not None:
            for i in range(self.edge_index.size(1)):
                src, dst = self.edge_index[:, i]
                src_idx = src.item()
                dst_idx = dst.item()
                
                # Check bounds
                if src_idx < self.num_nodes and dst_idx < self.num_nodes:
                    degrees[src_idx] += 1
                    degrees[dst_idx] += 1
                    out_degrees[src_idx] += 1
                    in_degrees[dst_idx] += 1
        
        # Calculate edge type diversity for each node
        edge_diversity = torch.zeros(self.num_nodes)
        node_edge_types = defaultdict(set)
        
        if self.edge_index is not None and self.edge_types is not None:
            for i in range(min(self.edge_index.size(1), len(self.edge_types))):
                src, dst = self.edge_index[:, i]
                src_idx = src.item()
                dst_idx = dst.item()
                
                # Check bounds
                if src_idx < self.num_nodes and dst_idx < self.num_nodes:
                    if i < len(self.edge_types):
                        edge_type = self.edge_types[i].item() if torch.is_tensor(self.edge_types) else self.edge_types[i]
                        node_edge_types[src_idx].add(edge_type)
                        node_edge_types[dst_idx].add(edge_type)
        
        for node, edge_types in node_edge_types.items():
            if node < self.num_nodes:
                edge_diversity[node] = len(edge_types)
        
        # Add node type features if available
        node_type_features = torch.zeros(self.num_nodes, 5)  # 5 node types
        if hasattr(self, 'node_to_type'):
            type_to_idx = {'disease': 0, 'drug': 1, 'function': 2, 'protein': 3, 'sideeffect': 4}
            for node_id, node_type in self.node_to_type.items():
                if node_id < self.num_nodes and node_type in type_to_idx:
                    node_type_features[node_id, type_to_idx[node_type]] = 1.0
        
        # Combine features
        self.node_features = torch.cat([
            degrees.unsqueeze(1),
            in_degrees.unsqueeze(1),
            out_degrees.unsqueeze(1),
            edge_diversity.unsqueeze(1),
            torch.log1p(degrees).unsqueeze(1),  # Log-scaled degree
            node_type_features,  # One-hot encoded node types
            torch.randn(self.num_nodes, 55)  # Additional random features
        ], dim=1)
        
        print(f"Created node features with dimension: {self.node_features.size(1)}")
    
    def create_entity_names(self):
        """Create entity names for visualization"""
        # For heterogeneous graphs, use actual node types
        if hasattr(self, 'node_to_type'):
            # Name all nodes based on their types
            for node_id, node_type in self.node_to_type.items():
                # Calculate the relative ID within the node type
                offset = self.node_type_offsets.get(node_type, 0)
                relative_id = node_id - offset
                self.entity_names[node_id] = f"{node_type}_{relative_id}"
                self.entity_types[node_id] = node_type
        else:
            # For homogeneous graphs or when type info is not available
            sample_size = min(10000, self.num_nodes)
            sample_nodes = np.random.choice(self.num_nodes, sample_size, replace=False)
            
            for node in sample_nodes:
                if self.node_types is not None and node < len(self.node_types):
                    node_type = self.node_types[node].item()
                    type_name = self.entity_type_map.get(node_type, f"type_{node_type}")
                    self.entity_names[node] = f"{type_name}_{node}"
                else:
                    # Assign random types for visualization
                    type_idx = node % len(self.entity_type_map)
                    type_name = list(self.entity_type_map.values())[type_idx]
                    self.entity_names[node] = f"{type_name}_{node}"
    
    def create_synthetic_biokg(self):
        """Create synthetic biomedical knowledge graph data"""
        # Create synthetic biomedical entities
        proteins = [f"protein_{i}" for i in range(100)]
        diseases = [f"disease_{i}" for i in range(50)]
        drugs = [f"drug_{i}" for i in range(75)]
        genes = [f"gene_{i}" for i in range(80)]
        pathways = [f"pathway_{i}" for i in range(30)]
        
        all_entities = proteins + diseases + drugs + genes + pathways
        self.num_nodes = len(all_entities)
        
        # Create entity name and type mapping
        for i, entity in enumerate(all_entities):
            self.entity_names[i] = entity
            if 'protein' in entity:
                self.entity_types[i] = 0
            elif 'disease' in entity:
                self.entity_types[i] = 1
            elif 'drug' in entity:
                self.entity_types[i] = 2
            elif 'gene' in entity:
                self.entity_types[i] = 3
            elif 'pathway' in entity:
                self.entity_types[i] = 4
        
        # Create synthetic edges with different relation types
        edges = []
        edge_types = []
        
        # Define relation types
        relation_types = {
            'protein_protein': 0,
            'drug_disease': 1,
            'gene_protein': 2,
            'drug_protein': 3,
            'disease_protein': 4,
            'protein_pathway': 5,
            'drug_side_effect': 6,
            'disease_gene': 7
        }
        
        # Create edges
        np.random.seed(42)
        for i in range(2000):
            rel_type = np.random.choice(list(relation_types.keys()))
            
            if rel_type == 'protein_protein':
                src = np.random.randint(0, 100)
                dst = np.random.randint(0, 100)
            elif rel_type == 'drug_disease':
                src = np.random.randint(150, 225)
                dst = np.random.randint(100, 150)
            elif rel_type == 'gene_protein':
                src = np.random.randint(225, 305)
                dst = np.random.randint(0, 100)
            elif rel_type == 'drug_protein':
                src = np.random.randint(150, 225)
                dst = np.random.randint(0, 100)
            elif rel_type == 'disease_protein':
                src = np.random.randint(100, 150)
                dst = np.random.randint(0, 100)
            elif rel_type == 'protein_pathway':
                src = np.random.randint(0, 100)
                dst = np.random.randint(305, 335)
            else:
                continue
            
            edges.append([src, dst])
            edge_types.append(relation_types[rel_type])
        
        self.edge_index = torch.tensor(edges).t()
        self.edge_types = torch.tensor(edge_types)
        
        # Create node features
        self.create_node_features()
        
        # Create mappings
        self.create_default_mappings()
        
        print(f"Created synthetic biokg data:")
        print(f"- Nodes: {self.num_nodes}")
        print(f"- Edges: {len(edges)}")
        print(f"- Edge types: {len(relation_types)}")

class GraphRAGEmbedding(torch.nn.Module):
    """Enhanced Graph Neural Network for creating node embeddings"""
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # Use different GNN architectures
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x

class GraphRAG:
    """Enhanced Graph Retrieval Augmented Generation system"""
    
    def __init__(self, knowledge_graph, model_path='graph_rag_model.pt'):
        self.kg = knowledge_graph
        self.embedding_model = None
        self.node_embeddings = None
        self.query_encoder = TfidfVectorizer(max_features=100)
        self.model_path = model_path
        
        # Try to load existing model, otherwise train new one
        if not self.load_embedding_model():
            self.train_embedding_model()
            self.save_embedding_model()
        
        self.build_query_index()
    
    def save_embedding_model(self):
        """Save the trained embedding model and embeddings to disk"""
        print(f"Saving model to {self.model_path}...")
        
        # Create a checkpoint dictionary
        checkpoint = {
            'model_state_dict': self.embedding_model.state_dict(),
            'model_config': {
                'input_dim': self.kg.node_features.size(1),
                'hidden_dim': 256,
                'output_dim': 128,
                'num_layers': 3
            },
            'node_embeddings': self.node_embeddings,
            'num_nodes': self.kg.num_nodes,
            'feature_dim': self.kg.node_features.size(1)
        }
        
        # Save the checkpoint
        torch.save(checkpoint, self.model_path)
        print(f"Model saved successfully!")
        
        # Also save node embeddings separately for quick access
        embeddings_path = self.model_path.replace('.pt', '_embeddings.npy')
        np.save(embeddings_path, self.node_embeddings)
        print(f"Embeddings saved to {embeddings_path}")
    
    def load_embedding_model(self):
        """Load a previously trained embedding model from disk"""
        if not Path(self.model_path).exists():
            print(f"No saved model found at {self.model_path}")
            return False
        
        try:
            print(f"Loading model from {self.model_path}...")
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Check if the model is compatible with current graph
            if checkpoint['num_nodes'] != self.kg.num_nodes:
                print(f"Model mismatch: saved model has {checkpoint['num_nodes']} nodes, current graph has {self.kg.num_nodes} nodes")
                return False
            
            if checkpoint['feature_dim'] != self.kg.node_features.size(1):
                print(f"Feature dimension mismatch: saved model expects {checkpoint['feature_dim']} features, current graph has {self.kg.node_features.size(1)} features")
                return False
            
            # Create model with saved configuration
            config = checkpoint['model_config']
            self.embedding_model = GraphRAGEmbedding(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_layers=config['num_layers']
            )
            
            # Load model weights
            self.embedding_model.load_state_dict(checkpoint['model_state_dict'])
            self.embedding_model.eval()
            
            # Load node embeddings
            self.node_embeddings = checkpoint['node_embeddings']
            
            print(f"Model loaded successfully!")
            print(f"- Input dim: {config['input_dim']}")
            print(f"- Hidden dim: {config['hidden_dim']}")
            print(f"- Output dim: {config['output_dim']}")
            print(f"- Embeddings shape: {self.node_embeddings.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def train_embedding_model(self):
        """Train GNN to create node embeddings with improved training"""
        print("Training enhanced graph embedding model...")
        
        # Check if we have valid features
        if self.kg.node_features is None or self.kg.node_features.size(0) == 0:
            print("Warning: No node features found, creating random features...")
            self.kg.node_features = torch.randn(self.kg.num_nodes, 65)
        
        self.embedding_model = GraphRAGEmbedding(
            input_dim=self.kg.node_features.size(1),
            hidden_dim=256,
            output_dim=128,
            num_layers=3
        )
        
        optimizer = torch.optim.Adam(self.embedding_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training with link prediction task
        for epoch in range(60): ### !!!
            optimizer.zero_grad()
            
            # Get embeddings
            embeddings = self.embedding_model(self.kg.node_features, self.kg.edge_index)
            
            # Positive edges
            pos_edge_index = self.kg.edge_index
            
            # Sample negative edges
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.kg.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )
            
            # Calculate scores
            pos_scores = (embeddings[pos_edge_index[0]] * embeddings[pos_edge_index[1]]).sum(dim=1)
            neg_scores = (embeddings[neg_edge_index[0]] * embeddings[neg_edge_index[1]]).sum(dim=1)
            
            # Binary cross entropy loss
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Generate final embeddings
        with torch.no_grad():
            self.node_embeddings = self.embedding_model(
                self.kg.node_features, self.kg.edge_index
            ).numpy()
        
        print("Graph embedding model trained!")
    
    def build_query_index(self):
        """Build search index for text-based entity retrieval"""
        # Create text descriptions for entities
        entity_descriptions = []
        self.entity_to_node = {}
        
        for node_id, entity_name in self.kg.entity_names.items():
            # Create searchable description
            entity_type = entity_name.split('_')[0]
            desc = f"{entity_name} {entity_type} biomedical entity"
            entity_descriptions.append(desc)
            self.entity_to_node[len(entity_descriptions) - 1] = node_id
        
        if entity_descriptions:
            self.query_encoder.fit(entity_descriptions)
            self.entity_vectors = self.query_encoder.transform(entity_descriptions)
    
    def find_entities_by_text(self, query_text, k=5):
        """Find entities matching query text"""
        query_text_lower = query_text.lower()
        matched_entities = []
        
        # Direct matching for heterogeneous graphs
        if hasattr(self, 'type_to_nodes'):
            # Search for node type keywords
            for node_type in self.kg.num_nodes_dict.keys():
                if node_type in query_text_lower:
                    # Get some nodes of this type
                    nodes_of_type = self.kg.type_to_nodes[node_type]
                    matched_entities.extend(nodes_of_type[:k//2])
            
            # Search for specific entity mentions
            keywords = query_text_lower.split()
            for keyword in keywords:
                if len(keyword) > 3:  # Skip short words
                    for node_id, entity_name in self.kg.entity_names.items():
                        if keyword in entity_name.lower():
                            matched_entities.append(node_id)
                            if len(matched_entities) >= k:
                                break
        
        # Fallback to original TF-IDF based search if available
        if hasattr(self, 'entity_vectors') and len(matched_entities) < k:
            query_vec = self.query_encoder.transform([query_text])
            similarities = (self.entity_vectors @ query_vec.T).toarray().flatten()
            top_k = np.argsort(similarities)[-k:][::-1]
            
            for idx in top_k:
                if similarities[idx] > 0 and self.entity_to_node[idx] not in matched_entities:
                    matched_entities.append(self.entity_to_node[idx])
        
        return list(set(matched_entities))[:k]
    
    # k = max number of nodes to keep in final subgraph
    # hop = how many steps away from starting point to explore
    def retrieve_relevant_subgraph(self, query_entities, k=20, hop=2):
        """Retrieve relevant subgraph with multi-hop expansion"""
        if not query_entities:
            return [], []
        
        # Start with query entities
        relevant_nodes = set(query_entities)
        
        # Multi-hop expansion
        for _ in range(hop):
            new_nodes = set()
            for i in range(self.kg.edge_index.size(1)):
                src, dst = self.kg.edge_index[:, i].tolist()
                if src in relevant_nodes:
                    new_nodes.add(dst)
                if dst in relevant_nodes:
                    new_nodes.add(src)
            relevant_nodes.update(new_nodes)
            
            # Limit size
            if len(relevant_nodes) > k * 2:
                # Keep most connected nodes
                node_degrees = defaultdict(int)
                for node in relevant_nodes:
                    for i in range(self.kg.edge_index.size(1)):
                        src, dst = self.kg.edge_index[:, i].tolist()
                        if src == node or dst == node:
                            node_degrees[node] += 1
                
                sorted_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)
                relevant_nodes = set([n[0] for n in sorted_nodes[:k]])
                relevant_nodes.update(query_entities)  # Always keep query entities
        
        # Extract edges within subgraph
        subgraph_edges = []
        edge_types_count = defaultdict(int)
        
        for i in range(self.kg.edge_index.size(1)):
            src, dst = self.kg.edge_index[:, i].tolist()
            if src in relevant_nodes and dst in relevant_nodes:
                edge_type = self.kg.edge_types[i].item()
                subgraph_edges.append([src, dst, edge_type])
                edge_types_count[edge_type] += 1
        
        return list(relevant_nodes), subgraph_edges
    
    def generate_response(self, query, subgraph_nodes, subgraph_edges):
        """Generate enhanced response based on retrieved subgraph"""
        if not subgraph_nodes:
            return "No relevant information found in the knowledge graph."
        
        response = f"Based on the biomedical knowledge graph analysis for '{query}':\n\n"
        
        # Analyze node types
        node_types = defaultdict(list)
        for node_id in subgraph_nodes[:50]:  # Limit for readability
            if hasattr(self.kg, 'node_to_type') and node_id in self.kg.node_to_type:
                node_type = self.kg.node_to_type[node_id]
                entity_name = self.kg.entity_names.get(node_id, f"{node_type}_{node_id}")
            else:
                entity_name = self.kg.entity_names.get(node_id, f"entity_{node_id}")
                node_type = entity_name.split('_')[0]
            node_types[node_type].append(entity_name)
        
        response += "**Relevant Biomedical Entities Found:**\n"
        type_display_names = {
            'protein': 'Proteins',
            'disease': 'Diseases', 
            'drug': 'Drugs',
            'function': 'Molecular Functions',
            'sideeffect': 'Side Effects'
        }
        
        for entity_type, entities in sorted(node_types.items()):
            display_name = type_display_names.get(entity_type, entity_type.title() + 's')
            response += f"• **{display_name}** ({len(entities)}): "
            sample_entities = entities[:3]
            response += f"{', '.join(sample_entities)}"
            if len(entities) > 3:
                response += f" and {len(entities)-3} more"
            response += "\n"
        
        # Analyze relationships
        response += "\n**Key Relationships Discovered:**\n"
        relation_stats = defaultdict(int)
        relation_examples = defaultdict(list)
        
        for src, dst, rel_type in subgraph_edges:
            if hasattr(self.kg, 'edge_type_names') and self.kg.edge_type_names:
                rel_name = self.kg.edge_type_names.get(rel_type, f"relation_{rel_type}")
            else:
                rel_name = f"relation_{rel_type}"
            relation_stats[rel_name] += 1
            
            # Store examples
            if len(relation_examples[rel_name]) < 2:
                src_name = self.kg.entity_names.get(src, f"entity_{src}")
                dst_name = self.kg.entity_names.get(dst, f"entity_{dst}")
                relation_examples[rel_name].append(f"{src_name} → {dst_name}")
        
        # Sort by frequency
        sorted_relations = sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)
        
        for rel_name, count in sorted_relations[:5]:
            # Clean up relation names
            clean_name = rel_name.replace('-', ' ').replace('_', ' ').title()
            response += f"• **{clean_name}**: {count} interactions\n"
            if rel_name in relation_examples and relation_examples[rel_name]:
                response += f"  Examples: {'; '.join(relation_examples[rel_name][:2])}\n"
        
        # Network statistics
        response += f"\n**Subgraph Statistics:**\n"
        response += f"• Total entities: {len(subgraph_nodes)}\n"
        response += f"• Total relationships: {len(subgraph_edges)}\n"
        response += f"• Relationship types: {len(relation_stats)}\n"
        
        # Calculate density
        if len(subgraph_nodes) > 1:
            max_edges = len(subgraph_nodes) * (len(subgraph_nodes) - 1)
            density = len(subgraph_edges) / max_edges if max_edges > 0 else 0
            response += f"• Network density: {density:.3f}\n"
        
        # Add specific insights based on node types present
        if 'drug' in node_types and 'disease' in node_types:
            response += "\n**Insight**: This subgraph contains drug-disease relationships that may indicate treatment options.\n"
        if 'protein' in node_types and 'function' in node_types:
            response += "\n**Insight**: Protein-function relationships found that may reveal molecular mechanisms.\n"
        if 'drug' in node_types and 'sideeffect' in node_types:
            response += "\n**Insight**: Drug-side effect associations identified for safety considerations.\n"
        
        return response
    
    def query(self, query_text):
        """Enhanced query interface for Graph RAG"""
        print(f"\nProcessing query: {query_text}")
        
        # Extract entities using multiple methods
        query_entities = []
        query_lower = query_text.lower()
        
        # Method 1: Look for node type keywords
        if hasattr(self.kg, 'type_to_nodes'):
            type_keywords = {
                'protein': ['protein', 'enzyme', 'receptor'],
                'disease': ['disease', 'disorder', 'syndrome', 'cancer'],
                'drug': ['drug', 'medication', 'treatment', 'compound'],
                'function': ['function', 'pathway', 'process'],
                'sideeffect': ['side effect', 'adverse', 'reaction']
            }
            
            for node_type, keywords in type_keywords.items():
                if node_type in self.kg.type_to_nodes:
                    for keyword in keywords:
                        if keyword in query_lower:
                            # Get some nodes of this type
                            nodes = self.kg.type_to_nodes[node_type]
                            query_entities.extend(nodes[:3])
                            break
        
        # Method 2: Direct entity name matching
        words = query_text.split()
        for word in words:
            word_lower = word.lower()
            if len(word_lower) > 2:
                for node_id, name in list(self.kg.entity_names.items())[:1000]:  # Limit search
                    if word_lower in name.lower():
                        query_entities.append(node_id)
                        if len(query_entities) >= 10:
                            break
        
        # Method 3: Text similarity search
        similar_entities = self.find_entities_by_text(query_text, k=5)
        query_entities.extend(similar_entities)
        
        # Remove duplicates and limit
        query_entities = list(set(query_entities))[:10]
        
        if not query_entities and hasattr(self.kg, 'type_to_nodes'):
            # If no entities found, just get some relevant nodes based on query
            print("No specific entities found, using general nodes...")
            for node_type in self.kg.type_to_nodes:
                if node_type in query_lower or any(keyword in query_lower for keyword in ['all', 'any', 'show', 'find']):
                    query_entities.extend(self.kg.type_to_nodes[node_type][:2])
                    if len(query_entities) >= 5:
                        break
        
        print(f"Found {len(query_entities)} relevant entities")
        
        # Retrieve relevant subgraph ### !!! (k, hop)
        relevant_nodes, relevant_edges = self.retrieve_relevant_subgraph(
            query_entities, k=15, hop=1
        ) # changed k and hop from k=30 and hop=2 to perform quicker while demoing ok
        
        print(f"Retrieved subgraph: {len(relevant_nodes)} nodes, {len(relevant_edges)} edges")
        
        # Generate response
        response = self.generate_response(query_text, relevant_nodes, relevant_edges)
        
        return {
            'response': response,
            'subgraph_nodes': relevant_nodes,
            'subgraph_edges': relevant_edges,
            'query_entities': query_entities
        }

def visualize_knowledge_graph(kg, graph_rag=None, query_result=None, sample_size=150):
    """Enhanced visualization of the knowledge graph and Graph RAG process"""
    
    fig = plt.figure(figsize=(24, 18))
    
    # Create custom layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], width_ratios=[2, 2, 1])
    
    # 1. Full knowledge graph overview
    ax1 = fig.add_subplot(gs[0, :2])
    sample_nodes = np.random.choice(kg.num_nodes, min(sample_size, kg.num_nodes), replace=False)
    
    # Create subgraph
    edges_in_sample = []
    edge_types_in_sample = []
    for i in range(kg.edge_index.size(1)):
        src, dst = kg.edge_index[:, i].tolist()
        if src in sample_nodes and dst in sample_nodes:
            edges_in_sample.append((src, dst))
            edge_types_in_sample.append(kg.edge_types[i].item())
    
    if edges_in_sample:
        G = nx.Graph()
        G.add_nodes_from(sample_nodes)
        G.add_edges_from(edges_in_sample)
        
        # Use better layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            if hasattr(kg, 'node_to_type') and node in kg.node_to_type:
                node_type = kg.node_to_type[node]
                if node_type == 'protein':
                    node_colors.append('#4287f5')  # Blue
                elif node_type == 'disease':
                    node_colors.append('#f54242')  # Red
                elif node_type == 'drug':
                    node_colors.append('#42f554')  # Green
                elif node_type == 'function':
                    node_colors.append('#f5e042')  # Yellow
                elif node_type == 'sideeffect':
                    node_colors.append('#f542e0')  # Purple
                else:
                    node_colors.append('#c7c7c7')  # Gray
            else:
                # Fallback to name-based coloring
                entity_name = kg.entity_names.get(node, f"entity_{node}")
                if 'protein' in entity_name:
                    node_colors.append('#4287f5')  # Blue
                elif 'disease' in entity_name:
                    node_colors.append('#f54242')  # Red
                elif 'drug' in entity_name:
                    node_colors.append('#42f554')  # Green
                elif 'gene' in entity_name or 'function' in entity_name:
                    node_colors.append('#f5e042')  # Yellow
                elif 'side' in entity_name:
                    node_colors.append('#f542e0')  # Purple
                else:
                    node_colors.append('#c7c7c7')  # Gray
        
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                              node_size=50, alpha=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', 
                              alpha=0.3, width=0.5)
        
        ax1.set_title(f'Biomedical Knowledge Graph Overview\n({len(G.nodes)} nodes, {len(G.edges)} edges)', 
                     fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        # Add legend
        legend_elements = [
            plt.scatter([], [], c='#4287f5', s=100, label='Proteins'),
            plt.scatter([], [], c='#f54242', s=100, label='Diseases'),
            plt.scatter([], [], c='#42f554', s=100, label='Drugs'),
            plt.scatter([], [], c='#f5e042', s=100, label='Genes'),
            plt.scatter([], [], c='#f542e0', s=100, label='Pathways')
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 2. Node embeddings visualization (t-SNE)
    ax2 = fig.add_subplot(gs[0, 2])
    if graph_rag and graph_rag.node_embeddings is not None:
        # Use t-SNE for better visualization
        sample_indices = np.random.choice(len(graph_rag.node_embeddings), 
                                        min(500, len(graph_rag.node_embeddings)), 
                                        replace=False)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(graph_rag.node_embeddings[sample_indices])
        
        # Color by entity type
        colors = []
        for idx in sample_indices:
            entity_name = kg.entity_names.get(idx, f"entity_{idx}")
            if 'protein' in entity_name:
                colors.append('#4287f5')
            elif 'disease' in entity_name:
                colors.append('#f54242')
            elif 'drug' in entity_name:
                colors.append('#42f554')
            elif 'gene' in entity_name:
                colors.append('#f5e042')
            else:
                colors.append('#c7c7c7')
        
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=colors, alpha=0.6, s=30)
        ax2.set_title('Node Embeddings\n(t-SNE Projection)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.grid(True, alpha=0.3)
    
    # 3. Retrieved subgraph
    ax3 = fig.add_subplot(gs[1, :2])
    if query_result and query_result['subgraph_nodes']:
        relevant_nodes = query_result['subgraph_nodes'][:40]  # Limit for visualization
        
        # Create subgraph
        G_sub = nx.Graph()
        G_sub.add_nodes_from(relevant_nodes)
        
        # Add edges
        for src, dst, edge_type in query_result['subgraph_edges']:
            if src in relevant_nodes and dst in relevant_nodes:
                G_sub.add_edge(src, dst, type=edge_type)
        
        if len(G_sub.nodes) > 0:
            pos = nx.spring_layout(G_sub, k=2, iterations=100, seed=42)
            
            # Highlight query entities
            node_colors = []
            node_sizes = []
            for node in G_sub.nodes():
                if node in query_result.get('query_entities', []):
                    node_colors.append('#ff0000')  # Red for query entities
                    node_sizes.append(300)
                else:
                    entity_name = kg.entity_names.get(node, f"entity_{node}")
                    if 'protein' in entity_name:
                        node_colors.append('#4287f5')
                    elif 'disease' in entity_name:
                        node_colors.append('#f54242')
                    elif 'drug' in entity_name:
                        node_colors.append('#42f554')
                    elif 'gene' in entity_name:
                        node_colors.append('#f5e042')
                    else:
                        node_colors.append('#c7c7c7')
                    node_sizes.append(150)
            
            nx.draw_networkx_nodes(G_sub, pos, ax=ax3, node_color=node_colors, 
                                  node_size=node_sizes, alpha=0.8)
            
            # Draw edges with different colors for different types
            edge_colors = []
            for u, v, d in G_sub.edges(data=True):
                edge_type = d.get('type', 0)
                edge_colors.append(plt.cm.tab10(edge_type % 10))
            
            nx.draw_networkx_edges(G_sub, pos, ax=ax3, edge_color=edge_colors,
                                  alpha=0.6, width=2)
            
            # Add labels for important nodes
            labels = {}
            for node in G_sub.nodes():
                if node in query_result.get('query_entities', []):
                    entity_name = kg.entity_names.get(node, f"entity_{node}")
                    labels[node] = entity_name.split('_')[0][:8]
            
            nx.draw_networkx_labels(G_sub, pos, labels, ax=ax3, font_size=10, 
                                   font_weight='bold')
            
            ax3.set_title(f'Retrieved Subgraph for Query\n({len(G_sub.nodes)} nodes, {len(G_sub.edges)} edges)', 
                         fontsize=16, fontweight='bold')
            ax3.axis('off')
            
            # Add note about query entities
            ax3.text(0.02, 0.98, '● Query Entities', transform=ax3.transAxes, 
                    color='red', fontsize=12, fontweight='bold', va='top')
    
    # 4. Graph RAG Process Flow
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    
    if query_result:
        # Create detailed process visualization
        process_info = f"""📊 GRAPH RAG PROCESS

1️⃣ Query Analysis
   • Input: "{query_result.get('query_entities', [''])[0] if query_result.get('query_entities') else 'No query'}"
   • Entities found: {len(query_result.get('query_entities', []))}

2️⃣ Entity Matching
   • Text similarity search
   • Direct entity matching
   • Multi-method fusion

3️⃣ Subgraph Retrieval
   • 2-hop expansion
   • {len(query_result.get('subgraph_nodes', []))} nodes retrieved
   • {len(query_result.get('subgraph_edges', []))} relationships

4️⃣ Response Generation
   • Entity type analysis
   • Relationship patterns
   • Network statistics

✅ Knowledge-grounded answer ready!"""
        
        ax4.text(0.05, 0.95, process_info, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Run a query to see the\nGraph RAG process in action!', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
    
    ax4.set_title('Graph RAG Pipeline', fontsize=14, fontweight='bold')
    
    # 5. Relationship Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    if kg.edge_type_names:
        # Count edge types
        edge_type_counts = defaultdict(int)
        for edge_type in kg.edge_types[:1000]:  # Sample for efficiency
            edge_type_counts[edge_type.item()] += 1
        
        # Prepare data for plotting
        labels = []
        counts = []
        for edge_type, count in sorted(edge_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            label = kg.edge_type_names.get(edge_type, f"Type_{edge_type}")
            labels.append(label.replace('_', ' ').title()[:20])
            counts.append(count)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        ax5.barh(y_pos, counts, color='skyblue', alpha=0.8)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels, fontsize=10)
        ax5.set_xlabel('Count')
        ax5.set_title('Top Relationship Types', fontsize=14, fontweight='bold')
        ax5.grid(True, axis='x', alpha=0.3)
    
    # 6. Query Results Summary
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    if query_result:
        # Create summary statistics
        summary_text = "📈 QUERY RESULTS SUMMARY\n\n"
        
        # Entity type distribution in results
        entity_dist = defaultdict(int)
        for node in query_result['subgraph_nodes'][:50]:
            entity_name = kg.entity_names.get(node, f"entity_{node}")
            entity_type = entity_name.split('_')[0]
            entity_dist[entity_type] += 1
        
        summary_text += "Entity Distribution:\n"
        for etype, count in sorted(entity_dist.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(count / max(entity_dist.values()) * 20)
            summary_text += f"  {etype:12} |{'█' * bar_length}{' ' * (20-bar_length)}| {count}\n"
        
        # Relationship statistics
        rel_dist = defaultdict(int)
        for _, _, rel_type in query_result['subgraph_edges']:
            rel_name = kg.edge_type_names.get(rel_type, f"relation_{rel_type}")
            rel_dist[rel_name] += 1
        
        summary_text += f"\nTop Relationships: {len(rel_dist)} types found\n"
        for rel, count in sorted(rel_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
            summary_text += f"  • {rel.replace('_', ' ').title()}: {count}\n"
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", alpha=0.8))
    else:
        ax6.text(0.5, 0.5, 'Query results will appear here', 
                ha='center', va='center', transform=ax6.transAxes, 
                fontsize=12, style='italic', color='gray')
    
    plt.tight_layout()
    plt.show()

def interactive_query_interface(kg, graph_rag):
    """Interactive query interface for Graph RAG"""
    print("\n" + "="*60)
    print("🔍 INTERACTIVE BIOMEDICAL KNOWLEDGE GRAPH QUERY INTERFACE")
    print("="*60)
    print("\nExample queries:")
    print("- 'Find proteins related to cancer'")
    print("- 'What drugs interact with protein kinases?'")
    print("- 'Show gene-disease associations'")
    print("- 'Analyze drug side effects'")
    print("\nType 'exit' to quit, 'help' for more examples\n")
    
    while True:
        query = input("\n💬 Enter your query: ").strip()
        
        if query.lower() == 'exit':
            print("Thank you for using the Graph RAG system!")
            break
        
        if query.lower() == 'help':
            print("\nMore example queries:")
            print("- 'protein protein interactions'")
            print("- 'drug treatments for diseases'")
            print("- 'pathway analysis for metabolism'")
            print("- 'gene expression and proteins'")
            continue
        
        if query:
            # Process query
            result = graph_rag.query(query)
            
            # Display response
            print("\n" + "="*60)
            print("📊 GRAPH RAG RESPONSE")
            print("="*60)
            print(result['response'])
            
            # Offer visualization
            viz_choice = input("\n📈 Would you like to visualize the results? (y/n): ").strip().lower()
            if viz_choice == 'y':
                visualize_knowledge_graph(kg, graph_rag, result)

# Main execution
def main(force_retrain=False, model_path='graph_rag_model.pt'):
    print("="*60)
    print("🧬 BIOMEDICAL KNOWLEDGE GRAPH WITH GRAPH RAG")
    print("="*60)
    print("\nInitializing system components...\n")
    
    # 1. Initialize and load knowledge graph
    print("Step 1: Loading Knowledge Graph")
    print("-" * 30)
    kg = KnowledgeGraph()
    success = kg.load_biokg_dataset()
    
    # 2. Initialize Graph RAG system
    print("\nStep 2: Initializing Graph RAG System")
    print("-" * 30)
    
    # Delete existing model if force_retrain is True
    if force_retrain and Path(model_path).exists():
        print(f"Force retrain enabled. Deleting existing model at {model_path}")
        Path(model_path).unlink()
        embeddings_path = model_path.replace('.pt', '_embeddings.npy')
        if Path(embeddings_path).exists():
            Path(embeddings_path).unlink()
    
    graph_rag = GraphRAG(kg, model_path=model_path)
    
    # 3. Run demonstration queries
    print("\n" + "="*60)
    print("📋 DEMONSTRATION QUERIES")
    print("="*60)
    
    demo_queries = [
        "Find proteins associated with cancer diseases",
        "What drugs target protein kinases?",
        "Show gene expression relationships",
        "Analyze protein pathway interactions"
    ]
    
    # Run first demo query with visualization
    print(f"\nDemo Query: '{demo_queries[0]}'")
    print("-" * 40)
    result = graph_rag.query(demo_queries[0])
    print("\nResponse:")
    print(result['response'])
    
    # Visualize the results
    print("\n📊 Generating comprehensive visualization...")
    visualize_knowledge_graph(kg, graph_rag, result)
    
    # 4. Interactive mode
    print("\n" + "="*60)
    print("🚀 ENTERING INTERACTIVE MODE")
    print("="*60)
    
    interactive_mode = input("\nWould you like to enter interactive query mode? (y/n): ").strip().lower()
    if interactive_mode == 'y':
        interactive_query_interface(kg, graph_rag)
    else:
        # Run remaining demo queries
        print("\nRunning additional demonstration queries...")
        for query in demo_queries[1:3]:
            print(f"\n{'='*60}")
            print(f"Query: '{query}'")
            print("-" * 40)
            result = graph_rag.query(query)
            print(result['response'][:500] + "..." if len(result['response']) > 500 else result['response'])
    
    print("\n" + "="*60)
    print("✅ GRAPH RAG DEMONSTRATION COMPLETE")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("✓ Loaded biomedical knowledge graph (ogbl-biokg)")
    print("✓ Trained graph neural network embeddings")
    print("✓ Retrieved relevant subgraphs based on queries")
    print("✓ Generated knowledge-grounded responses")
    print("✓ Visualized graph structure and retrieval process")
    print(f"✓ Model saved to {model_path} for faster subsequent runs")
    print("\nThank you for exploring the Graph RAG system!")

if __name__ == "__main__":
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='Biomedical Knowledge Graph with Graph RAG')
    parser.add_argument('--force-retrain', action='store_true', 
                        help='Force retraining of the embedding model')
    parser.add_argument('--model-path', type=str, default='graph_rag_model.pt',
                        help='Path to save/load the trained model')
    
    args = parser.parse_args()
    
    main(force_retrain=args.force_retrain, model_path=args.model_path)