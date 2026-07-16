import pyterrier as pt


def load_query_processor(input_data):
    """
       Creates a query processor using TIRA's PyTerrier integration to transform queries.

       Parameters
       ----------
       input_data : Path-like object
           Path to the input data that contains the queries to be processed.
           The path will be resolved to its absolute path before processing.

       Returns
       -------
       callable
           A query transformer function that processes queries using TIRA's PyTerrier
           integration. The transformed queries can be used for further information
           retrieval tasks.

       Examples
       --------
       >>> from pathlib import Path
       >>> input_path = Path("queries.json")
       >>> processor = load_query_processor(input_path)
       >>> transformed_queries = processor(input_queries)
       """
    from tira.pyterrier_integration import PyTerrierIntegration as pt
    from tira.rest_api_client import Client
    return pt(Client()).transform_queries(str(input_data.resolve().absolute()), None)

def ir_datasets_id_is_supported(ir_datasets_id):
    from tira.tirex import IRDS_TO_TIREX_DATASET
    return ir_datasets_id in IRDS_TO_TIREX_DATASET

def load_run(ir_datasets_id, approach):
    if not ir_datasets_id_is_supported(ir_datasets_id):
        raise ValueError(f"The ir_datasets_id '{ir_datasets_id}' is not supported.")

    return pt.Artifact.from_url(f"tira:{ir_datasets_id}/{approach}")