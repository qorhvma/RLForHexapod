from pathlib import Path
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "RLforHexapod" / "assets" / "obstacles"


def make_stairs(path: Path, step_h: float, n_steps: int = 3, step_d: float = 0.30,
                width: float = 1.0, top_len: float = 0.8):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Stairs")
    stage.SetDefaultPrim(root.GetPrim())

    # 마찰 재질
    mat = UsdShade.Material.Define(stage, "/Stairs/PhysMat")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(1.0)
    pm.CreateDynamicFrictionAttr(1.0)
    pm.CreateRestitutionAttr(0.0)

    for i in range(n_steps):
        h = step_h * (i + 1)                       # 바닥부터 채운 블록 (틈 없음)
        depth = step_d if i < n_steps - 1 else top_len
        cube = UsdGeom.Cube.Define(stage, f"/Stairs/step_{i}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(i * step_d + depth / 2, 0.0, h / 2))
        xf.SetScale(Gf.Vec3f(depth, width, h))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())   # RigidBodyAPI 없음 → 정적 충돌체
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    stage.GetRootLayer().Save()
    print("저장:", path)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for h_mm in (40, 80, 120, 160):   # 커리큘럼용으로 여러 높이
        make_stairs(OUT_DIR / f"stairs_{h_mm}mm.usda", h_mm / 1000)
