"""Small synthetic checks only; no simulator or dependency installation."""
from .detectors import Observation as O, PickDetector, PlaceDetector

def main():
    p = PickDetector('bowl',0,0.8)
    assert p.update(O(0,'bowl',0.86,False)).status == 'in_progress'
    for i in range(1,12):
        r = p.update(O(i*.05,'bowl',0.86,True))
    assert r.status == 'succeeded'
    q = PickDetector('bowl',0,0.8)
    q.update(O(0,'bowl',0.86,True))
    assert q.update(O(.05,'bowl',0.8,False)).reason == 'grasp_lost'
    assert q.update(O(.1,'bowl')).status == 'unknown'
    assert q.update(O(10,'bowl')).reason == 'attempt_timeout'
    d = PlaceDetector('bowl','plate',0)
    assert d.update(O(0,'bowl',target_grasped=True,destination='plate',on_support=True,
        released=False,linear_speed_m_s=0,angular_speed_rad_s=0)).status == 'in_progress'
    for i in range(1,12):
        r = d.update(O(i*.05,'bowl',target_grasped=False,destination='plate',on_support=True,
            released=True,linear_speed_m_s=0,angular_speed_rad_s=0))
    assert r.status == 'succeeded'
    d = PlaceDetector('bowl','plate',0)
    assert d.update(O(0,'bowl',destination='plate')).status == 'unknown'
    p = PickDetector('bowl',0,.8)
    p.update(O(0,'bowl',.86,True))
    assert p.update(O(.6,'bowl',.86,True)).status == 'in_progress'
    try:
        p.update(O(.6,'bowl',.86,True))
        raise AssertionError('duplicate timestamp accepted')
    except ValueError:
        pass
    print('Synthetic checks passed: pick/place persistence, grasp loss, missing evidence, timeout, gaps, timestamps.')

if __name__ == '__main__':
    main()
