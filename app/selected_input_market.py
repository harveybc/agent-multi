"""Selected recipe projection without traversing a historical narrower recipe."""
from app.recipe_runtime import RecipeMarket
from app.rl_subset import project_inputs


def market_for(root,spec,selected_recipe):
    paths={k:str(root/v) for k,v in spec['inputs'].items()}
    paths['recipe']=str(root/'SUPPORT_RECIPE.json')
    market=RecipeMarket(dict(inputs=paths,engine_root=str(root/'engine'),
        recipe_sha256=spec['files']['SUPPORT_RECIPE.json']))
    market.inputs=project_inputs(market.inputs,selected_recipe)
    context=spec['effective']['architecture'].get('long_context')
    if context is not None:
        from app.multiscale_context import pack_context
        market.inputs=pack_context(market.inputs,context)
    market.recipe=selected_recipe
    return market
