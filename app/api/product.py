from fastapi import APIRouter, Request, Response

router = APIRouter(
    prefix="/internal/guide",
    tags=["导购商品"],
)


@router.post("/products", status_code=204)
async def submit_product(request: Request):
    data = await request.json()

    await request.app.state.product_service.submit_product(
        product_id=data["productId"],
        name=data["name"],
        model=data.get("model"),
        cover_url=data.get("coverUrl"),
        price=data.get("price"),
    )

    return Response(status_code=204)
