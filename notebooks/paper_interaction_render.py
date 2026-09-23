"""Offline presentation views of recorded states; never substitutes for agent captures."""
from pathlib import Path
import json, math, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from assembly_world_agent.episode_io import read_episode
from assembly_world_agent.episode_model import compiled_parts
from assembly_world_agent.vis.replay import StateRenderer
from paper_interaction import ROOT, PAPER

WIDTH, HEIGHT = 1024, 768
BG = [252, 253, 254]


def unit(x):
    return x / np.linalg.norm(x)


def geometry(renderer, parts, state):
    mj=renderer.mj
    mj.mj_setState(renderer.model,renderer.data,np.asarray(state['integration']),renderer.spec)
    mj.mj_forward(renderer.model,renderer.data)
    result={}
    for p in parts:
        body=mj.mj_name2id(renderer.model,mj.mjtObj.mjOBJ_BODY,p['id'])
        v=np.asarray(p['geometry']['vertices'])
        result[p['id']]=v@renderer.data.xmat[body].reshape(3,3).T+renderer.data.xpos[body]
    return result


def render(renderer, vertices, target, direction, radius, agent_camera=None, grid=True):
    mj=renderer.mj; renderer.renderer.update_scene(renderer.data)
    scene=renderer.renderer.scene
    forward=-unit(direction);right=unit(np.cross(forward,[0,0,1]));up=np.cross(right,forward)
    near=.01
    allv=np.concatenate(list(vertices.values()))
    relative=(np.vstack([allv,np.array(agent_camera["position"])]) if agent_camera else allv)-target
    # Preserve the recorded viewing direction and perspective, fitting only distance.
    tangent=math.tan(math.radians(19))
    depth=relative@unit(direction)
    distance=float(max(np.max(depth+np.abs(relative@right)/(tangent*WIDTH/HEIGHT)),
                       np.max(depth+np.abs(relative@up)/tangent)))*1.15
    distance=max(distance,radius*2)
    if not grid and not agent_camera:distance=radius*1.2/tangent
    eye=target+unit(direction)*distance
    half=near*tangent
    for camera in scene.camera:
        camera.pos[:]=eye;camera.forward[:]=forward;camera.up[:]=up;camera.orthographic=0
        camera.frustum_near=near;camera.frustum_far=max(100,distance*10)
        camera.frustum_top=half;camera.frustum_bottom=-half;camera.frustum_center=0;camera.frustum_width=0
    for light in scene.lights[:scene.nlight]:
        if light.headlight:light.pos[:]=eye;light.dir[:]=forward
    scene.flags[mj.mjtRndFlag.mjRND_CULL_FACE]=False
    def line(a,b,color,width=.0015):
        if scene.ngeom>=scene.maxgeom:raise RuntimeError('Renderer geometry capacity exceeded')
        g=scene.geoms[scene.ngeom]
        mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_LINE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([*color,1.]))
        mj.mjv_connector(g,mj.mjtGeom.mjGEOM_LINE,2.0 if width<=.0015 else 4.0,np.asarray(a,float),np.asarray(b,float))
        g.emission=.7;g.specular=0;g.shininess=0
        g.category=mj.mjtCatBit.mjCAT_DECOR;scene.ngeom+=1
    def project(points):
        d=np.asarray(points)-target
        z=distance-d@unit(direction)
        return np.column_stack([WIDTH/2+(d@right)/z*HEIGHT/(2*tangent),HEIGHT/2-(d@up)/z*HEIGHT/(2*tangent)])
    labels=[];grid_lines=[]
    if grid:
        allv=np.concatenate(list(vertices.values()))
        z=min(0,float(allv[:,2].min()))-.012*radius
        step=10**math.floor(math.log10(max(radius,.001)))
        while radius/step>2:step*=2
        step*=.5
        center=np.floor(target[:2]/step)*step;extent=step*16
        for n in range(-16,17):
            q=n*step
            grid_lines.append(([center[0]+q,center[1]-extent,z],[center[0]+q,center[1]+extent,z]))
            grid_lines.append(([center[0]-extent,center[1]+q,z],[center[0]+extent,center[1]+q,z]))
    if agent_camera:
        pos=np.array(agent_camera['position']);look=np.array(agent_camera['target']);f=unit(look-pos)
        rr=np.cross(f,[0,0,1]);rr=unit(rr) if np.linalg.norm(rr)>1e-7 else np.array([1.,0,0]);uu=np.cross(rr,f)
        length=radius*.22;h=length*math.tan(math.radians(19));w=h*WIDTH/HEIGHT
        corners=[pos+f*length+rr*w*a+uu*h*b for a,b in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        for i,c in enumerate(corners):line(pos,c,[.77,.43,.14],.003);line(c,corners[(i+1)%4],[.77,.43,.14],.003)
        # Dashed optical axis through the recorded camera target.
        for t in np.arange(.08,.98,.09):line(pos+(look-pos)*t,pos+(look-pos)*(t+.035),[.77,.43,.14],.0015)
        labels.append((project([pos])[0]+[8,-18],'Camera',(168,91,31)))
    pixels=renderer.renderer.render()
    background=Image.new('RGB',(WIDTH*2,HEIGHT*2),tuple(BG))
    painter=ImageDraw.Draw(background)
    for a,b in grid_lines:
        # Clip grid segments to the perspective near plane before projection.
        a=np.array(a);b=np.array(b)
        za=distance-(a-target)@unit(direction);zb=distance-(b-target)@unit(direction)
        if max(za,zb)<=near:continue
        if za<=near:a=a+(b-a)*(near-za)/(zb-za)
        elif zb<=near:b=b+(a-b)*(near-zb)/(za-zb)
        uv=project([a,b])*2
        painter.line([tuple(uv[0]),tuple(uv[1])],fill=(225,230,235),width=2)
    composed=np.asarray(background).copy()
    foreground=np.any(pixels!=0,axis=2)
    composed[foreground]=pixels[foreground]
    image=Image.fromarray(composed).resize((WIDTH,HEIGHT),Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',25)
    for point,label,color in labels:
        draw.text(tuple(point),label,font=font,fill=color,stroke_width=1,stroke_fill=tuple(BG))
    if grid:
        # Screen-space orientation gizmo follows the recorded camera basis.
        origin=np.array([WIDTH-92.,90.])
        for axis,label,color in [(np.array([1,0,0]),'X',(177,76,70)),(np.array([0,1,0]),'Y',(72,137,100)),(np.array([0,0,1]),'Z',(74,118,169))]:
            delta=np.array([axis@right,-axis@up])*47
            end=origin+delta
            draw.line([tuple(origin),tuple(end)],fill=color,width=3)
            draw.ellipse((end[0]-3,end[1]-3,end[0]+3,end[1]+3),fill=color)
            offset=delta/max(np.linalg.norm(delta),1)*16 if np.linalg.norm(delta)>6 else np.array([-17.,15.])
            draw.text(tuple(end+offset),label,font=font,fill=color,anchor='mm')
    return image,project


def generate():
    out=PAPER/'fig/interaction';manifest=json.loads((out/'selection.json').read_text())
    rows=[manifest,*manifest['additional_rows']]
    metadata={'kind':'Offline re-rendering of recorded states, not agent observations', 'background_rgb':BG,
              'grid':'Display ground plane at min(0, lowest vertex) with scale-adaptive world XY grid',
              'axes':'World XYZ directions, translated to a display anchor; not a new physical origin',
              'camera':'Recorded position, target and 38-degree vertical field of view; frustum depth is illustrative',
              'framing':'Each recorded camera direction and 38-degree perspective, with centered object framing',
              'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':[]}
    for row in rows:
        archive=ROOT/'results'/row['source_archive'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
        ep=read_episode(archive);renderer=StateRenderer(ep)
        renderer.renderer.close()
        renderer.model.vis.global_.offwidth=2048
        renderer.model.vis.global_.offheight=1536
        renderer.renderer=renderer.mj.Renderer(renderer.model,height=1536,width=2048)
        parts=compiled_parts(renderer.model,ep['manifest']['objects'])
        final=row['panels'][-1]['camera'];direction=unit(np.array(final['position'])-np.array(final['target']))

        data={'sample_id':row['sample_id'],'archive_sha256':row['archive_sha256'],'direction':direction.tolist(),'panels':[]}
        geometries=[geometry(renderer,parts,ep['states'][p['state_index']]) for p in row['panels']]
        # Shared framing for the before/after IKEA correction pair.
        shared=None
        if row['dataset_label']=='IKEA-Manual':
            vv=np.concatenate([geometries[i]['part-0003'] for i in [2,3]])
            shared=((vv.min(0)+vv.max(0))/2,float(np.linalg.norm(vv.max(0)-vv.min(0))*.55))
        for idx,panel in enumerate(row['panels']):
            vertices=geometry(renderer,parts,ep['states'][panel['state_index']]);allv=np.concatenate(list(vertices.values()))
            item={'call_number':panel['call_number'],'state_index':panel['state_index'],'variants':{}}
            panel_direction=unit(np.array(panel['camera']['position'])-np.array(panel['camera']['target']))
            for camera_on in [False,True]:
                bounds=allv
                center=(bounds.min(0)+bounds.max(0))/2
                f=-unit(panel_direction);rr=unit(np.cross(f,[0,0,1]));uu=np.cross(rr,f)
                xx=(bounds-center)@rr;yy=(bounds-center)@uu
                center=center+rr*(xx.max()+xx.min())/2+uu*(yy.max()+yy.min())/2
                radius=float(max(np.ptp(yy)/2,np.ptp(xx)/2/(WIDTH/HEIGHT)))*1.08
                if shared and idx in [2,3]:
                    radius=max(radius,float(np.linalg.norm(bounds.max(0)-bounds.min(0))*.52))
                tag='camera' if camera_on else 'clean';prefix=row['dataset_label'].lower().replace(' ','-')
                name=f'assets/render-{prefix}-{panel["call_number"]}-{tag}.png'
                img,project=render(renderer,vertices,center,panel_direction,radius)
                if camera_on:
                    # Keep the main object at the same size; show the real camera in a context inset.
                    context=np.vstack([allv,np.array(panel['camera']['position'])])
                    c=(context.min(0)+context.max(0))/2
                    cr=float(np.linalg.norm(context.max(0)-context.min(0))*.55)
                    inset,_=render(renderer,vertices,c,unit(panel_direction+np.array([.8,-.5,.8])),cr,panel['camera'],grid=False)
                    inset.thumbnail((260,195),Image.Resampling.LANCZOS)
                    img.paste(inset,(12,HEIGHT-207))
                img.save(out/name)
                entry={'image':name,'center':center.tolist(),'radius':radius,'view_direction':panel_direction.tolist(),'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
                if shared and idx in [2,3]:
                    uv=project(vertices['part-0003']);entry['detail_box_xywh']=[*uv.min(0).tolist(),*(uv.max(0)-uv.min(0)).tolist()]
                item['variants'][tag]=entry
            if shared and idx in [2,3]:
                name=f'assets/detail-{panel["call_number"]}.png'
                img,_=render(renderer,vertices,*shared[:1],panel_direction,shared[1],grid=False)
                img.save(out/name);item['detail_image']=name
            data['panels'].append(item)
        renderer.close();metadata['rows'].append(data)
        print(row['dataset_label'],'rendered',flush=True)
    (out/'rendered-selection.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

if __name__=='__main__':generate()
