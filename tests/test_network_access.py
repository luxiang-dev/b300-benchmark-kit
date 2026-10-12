"""Local HTTP fixtures only; never publishes a real server or executes hardware."""
import base64
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

import dashboard
import network_access as access
import test_dashboard as fixtures


class NetworkPolicyTests(unittest.TestCase):
    def test_remote_host_and_local_only_boundaries(self):
        for host in ('198.51.100.8:19000', 'bench.example.com', 'server.lan:8765', '[2001:db8::1]:8765'):
            self.assertTrue(access.AccessPolicy(remote=True).host_ok(host,8765))
            self.assertFalse(access.AccessPolicy().host_ok(host,8765))
        self.assertTrue(access.AccessPolicy().host_ok('localhost:8765',8765))

    def test_invalid_hosts_and_origins_are_rejected(self):
        for value in ('', None, 'bad\nHost', 'name@host', 'host/path', 'host:99999', 'host\\path', 'http://host'):
            self.assertIsNone(access.authority(value))
        policy=access.AccessPolicy(remote=True)
        for value in ('null','https://other.example','https://bench.example/path','https://x@bench.example','https://bench.example?x=1','ftp://bench.example'):
            self.assertFalse(policy.origin_ok(value,'bench.example'))
        self.assertTrue(policy.origin_ok('https://BENCH.example:443','bench.example'))

    def test_proxy_rewritten_host_needs_explicit_public_origin(self):
        self.assertFalse(access.AccessPolicy(remote=True).origin_ok('https://bench.example','localhost:8765'))
        policy=access.AccessPolicy(remote=True,public_origins=['https://bench.example'])
        self.assertTrue(policy.origin_ok('https://bench.example','localhost:8765'))
        self.assertFalse(policy.origin_ok('http://bench.example','localhost:8765'))
        self.assertFalse(policy.origin_ok('https://attacker.example','localhost:8765'))

    def test_allowlist_does_not_accept_arbitrary_hosts(self):
        policy=access.AccessPolicy(remote=True,allowed_hosts=['bench.example','server.lan:8765'])
        self.assertTrue(policy.host_ok('BENCH.example',8765))
        self.assertFalse(policy.host_ok('other.example',8765))
        with self.assertRaises(ValueError):
            access.AccessPolicy(allowed_hosts=['https://bench.example'])
        with self.assertRaises(ValueError):
            access.AccessPolicy(public_origins=['https://bench.example/path'])

    def test_token_file_shape_and_permissions(self):
        with fixtures.temporary_workspace() as tmp:
            path=Path(tmp)/'private-token'
            path.write_text('synthetic_only_'+'x'*32,encoding='ascii')
            path.chmod(0o600)
            self.assertEqual(access.load_token(path),'synthetic_only_'+'x'*32)
            path.write_text('short',encoding='ascii')
            with self.assertRaises(ValueError):
                access.load_token(path)
            if os.name=='posix':
                path.chmod(0o644)
                with self.assertRaises(ValueError):
                    access.load_token(path)

    def test_default_listener_and_explicit_local_listener(self):
        for extra,host in [([], '0.0.0.0'),(['--host','127.0.0.1'],'127.0.0.1')]:
            with patch.object(dashboard.sys,'argv',['dashboard.py','--preview',*extra]), patch.object(dashboard,'JobManager'), patch.object(dashboard,'ThreadingHTTPServer') as factory, patch.object(dashboard.signal,'signal'):
                factory.return_value.serve_forever.side_effect=KeyboardInterrupt
                dashboard.main()
                self.assertEqual(factory.call_args.args[0],(host,8765))


class RemoteHTTPTests(unittest.TestCase):
    request=fixtures.DashboardHTTPTests.request
    add_job=fixtures.DashboardHTTPTests.add_job
    tearDown=fixtures.DashboardHTTPTests.tearDown

    def setUp(self):
        self.temporary=fixtures.temporary_workspace()
        self.root=Path(self.temporary.name).resolve()
        self.manager=dashboard.JobManager(self.root/'runs',preview=True)
        self.ready_patch=patch.object(self.manager,'ready',return_value=fixtures.readiness_fixture())
        self.ready_patch.start()
        self.policy=access.AccessPolicy(remote=True)
        self.server=dashboard.ThreadingHTTPServer(('127.0.0.1',0),dashboard.make_handler(self.manager,self.policy))
        self.server.daemon_threads=True
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.05},daemon=True)
        self.thread.start()

    def test_remote_mapping_can_read_page_and_real_state(self):
        for endpoint in ('/','/app.js','/style.css','/api/session','/api/state'):
            status,_,raw=self.request('GET',endpoint,headers={'Host':'bench.example:19443'})
            self.assertEqual(status,200)
        self.assertEqual(json.loads(raw)['readiness']['gpus'],[])

    def test_mapping_start_and_stop_keep_csrf_and_origin_protection(self):
        headers={'Host':'bench.example','Origin':'https://bench.example'}
        with patch.object(self.manager,'start',return_value='synthetic-only') as start:
            status,_,_=self.request('POST','/api/jobs',dict(profile='server'),headers=headers,authenticated=True)
            self.assertEqual(status,202)
            start.assert_called_once()
        with patch.object(self.manager,'stop') as stop:
            status,_,_=self.request('POST','/api/jobs/synthetic-only/stop',{},headers=headers,authenticated=True)
            self.assertEqual(status,202)
            stop.assert_called_once_with('synthetic-only')
        for supplied in ({'Host':'bench.example','Origin':'https://attacker.example','X-CSRF-Token':self.manager.token},headers):
            with patch.object(self.manager,'start') as start:
                status,_,_=self.request('POST','/api/jobs',dict(profile='server'),headers=supplied)
                self.assertEqual(status,403)
                start.assert_not_called()

    def test_explicit_origin_supports_host_rewrite_but_not_forwarded_spoof(self):
        headers={'Host':f'127.0.0.1:{self.port}','Origin':'https://bench.example'}
        with patch.object(self.manager,'start',return_value='synthetic-only'):
            self.assertEqual(self.request('POST','/api/jobs',dict(profile='server'),headers=headers,authenticated=True)[0],403)
            self.policy.public_origins=(access.AccessPolicy.origin_key('https://bench.example'),)
            self.assertEqual(self.request('POST','/api/jobs',dict(profile='server'),headers=headers,authenticated=True)[0],202)
            headers.update(Origin='https://attacker.example',**{'X-Forwarded-Host':'bench.example'})
            self.assertEqual(self.request('POST','/api/jobs',dict(profile='server'),headers=headers,authenticated=True)[0],403)

    def test_optional_login_protects_page_csrf_reports_and_mutations(self):
        self.policy.token='synthetic_only_'+'x'*32
        key,out=self.add_job()
        (out/'report.html').write_text('synthetic private report',encoding='utf-8')
        for endpoint in ('/','/api/session','/api/state',f'/runs/{key}/report.html',f'/api/jobs/{key}/log'):
            status,headers,_=self.request('GET',endpoint)
            self.assertEqual(status,401)
            self.assertIn('Basic',headers['WWW-Authenticate'])
        for value in ('Basic !!!','Bearer wrong','Basic '+base64.b64encode(b'operator:wrong').decode()):
            self.assertEqual(self.request('GET','/api/session',headers={'Authorization':value})[0],401)
        authorization='Basic '+base64.b64encode(('operator:'+self.policy.token).encode()).decode()
        self.assertEqual(self.request('GET','/api/session',headers={'Authorization':authorization})[0],200)
        self.assertEqual(self.request('POST','/api/jobs',dict(profile='server'),authenticated=True)[0],401)
        with patch.object(self.manager,'start',return_value='synthetic-only'):
            self.assertEqual(self.request('POST','/api/jobs',dict(profile='server'),headers={'Authorization':authorization},authenticated=True)[0],202)
